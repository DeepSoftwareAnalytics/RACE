# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
import copy
import sys
# from models.RvNNRvNNASTCodeAttn import BatchASTEncoder
import logging
from util import REPLACE, REPLACE_OLD, REPLACE_NEW,REPLACE_END,INSERT,INSERT_OLD,INSERT_NEW ,INSERT_END,DELETE,DELETE_END,KEEP,KEEP_END
from transformers import T5ForConditionalGeneration
logger = logging.getLogger(__name__)




class ECMGModel(T5ForConditionalGeneration):
    def __init__(self, base_model,config,args=None,sos_id=None, eos_id=None):
        super().__init__(config)
        # self.base_model = base_model
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.lm_head = base_model.lm_head
        self.pooler = nn.Sequential(nn.Linear(config.d_model,config.d_model ), nn.Tanh(),  nn.Dropout(0.5) )
        self.W_sim = nn.Linear(2 * config.d_model, 1)
        self.W_c = nn.Linear(config.d_model, config.d_model)
        self.args=args
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))

        self.beam_size=args.beam_size
        self.max_length=args.max_target_length
        self.sos_id=sos_id
        self.eos_id=eos_id

        self.lsm = nn.LogSoftmax(dim=-1)

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None,
     retrieved_source_ids=None, retrieved_source_mask=None, retrieved_target_ids=None, retrieved_target_mask=None,use_cache=None,return_dict=None
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs_of_retrieved_target_ids = self.encoder(input_ids=retrieved_target_ids ,attention_mask=retrieved_target_mask).last_hidden_state 
        
        bs = source_ids.shape[0]
        inputs = torch.cat((source_ids , retrieved_source_ids, ), 0)
        inputs_mask = torch.cat((source_mask , retrieved_source_mask), 0)
        # outputs 是个二元组 [0]--> [bs, sequence_len, dim]; [1] --> [bs, dim]
        encoder_outputs = self.encoder(input_ids=inputs, attention_mask=inputs_mask)
        outputs = encoder_outputs.last_hidden_state  # [bs*2,seq_len, dim]
        encoder_outputs_of_input_source_ids = outputs[:bs] # [bs,seq_len,dim]
        encoder_outputs_of_retrieved_source_ids = outputs[bs:]  # [bs,seq_len,dim]        

        input_code_representation = encoder_outputs_of_input_source_ids.mean(1) # [bs, model_dim]
        similar_code_representation = encoder_outputs_of_retrieved_source_ids.mean(1) # [bs, model_dim]
        cat_two_code_outputs = torch.cat((input_code_representation, similar_code_representation), dim=-1)
        sim = F.sigmoid(self.W_sim(cat_two_code_outputs))   # [batch_size, 1, 1]
        sim = sim.reshape(-1, 1, 1)  # [batch_size, 1, 1]
        
        # combine the input and retrieved result 
        combined_encoder_output = torch.cat((self.W_c(encoder_outputs_of_input_source_ids) * (1 - sim) ,  encoder_outputs_of_retrieved_target_ids* sim),dim=1)
        combined_encoder_mask = torch.cat((source_mask,retrieved_target_mask),dim=1)

        if target_ids is not None:
            decoder_outputs = self.decoder(
                input_ids=target_ids , # [bs, length]
                attention_mask=target_mask, # [bs, length]
                inputs_embeds=None,
                past_key_values=None,
                encoder_hidden_states=combined_encoder_output,
                encoder_attention_mask=combined_encoder_mask,
                head_mask=None,
                cross_attn_head_mask=None,
                use_cache=use_cache,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
            )
            sequence_output = decoder_outputs[0] # [bs, length, dim]
            lm_logits = self.lm_head(sequence_output)
    
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1 #[bs * (seq-1)]
            shift_logits = lm_logits[..., :-1, :].contiguous()  #[bs, seq_length-1,vocab_size]
            shift_labels = target_ids[..., 1:].contiguous() #[bs * (seq-1)]
          
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss,loss*active_loss.sum(),active_loss.sum()
            return outputs
            # return outputs
        else:
            preds=[]       
            zero=torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]): 
                context=combined_encoder_output[i:i+1] # [1,seq_len,  dim ]
                context_mask=combined_encoder_mask[i:i+1,:]
                beam = Beam(self.beam_size,self.sos_id,self.eos_id)
                input_ids=beam.getCurrentState() # [bs , 1 ]
                context=context.repeat(self.beam_size, 1, 1) # [beam_size,seq_len,  dim ]
                context_mask=context_mask.repeat(self.beam_size,1) # [beam_size, seq_len]
                for _ in range(self.max_length): 
                    if beam.done():
                        break
                    decoder_outputs = self.decoder(
                        input_ids=input_ids, # [bs, length]
                        attention_mask=None, # [bs, length]
                        inputs_embeds=None,
                        past_key_values=None,
                        encoder_hidden_states=context,
                        encoder_attention_mask=context_mask,
                        head_mask=None,
                        cross_attn_head_mask=None,
                        use_cache=use_cache,
                        output_attentions=None,
                        output_hidden_states=None,
                        return_dict=return_dict,
                    )
                    hidden_states=decoder_outputs[0][:,-1,:] #[beam_size, dim]
                    out = self.lsm(self.lm_head(hidden_states)).data #[beam_size, vocab_size]
                    beam.advance(out) 
                    #https://blog.csdn.net/kdongyi/article/details/103099589
                    # copy the choose beam and expand it 
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))  
                    input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
                hyp= beam.getHyp(beam.getFinal())
                pred=beam.buildTargetTokens(hyp)[:self.beam_size]
                pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
                preds.append(torch.cat(pred,0).unsqueeze(0))
    
            preds=torch.cat(preds,0)                
            return preds 
        



class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos # sos 1, eos 2
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1) # wordLk [beam_size. vocab_size]

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True) # beam size

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords # divide and floor
        self.prevKs.append(prevK) # which beam
        self.nextYs.append((bestScoresId - prevK * numWords)) # which word


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
        
