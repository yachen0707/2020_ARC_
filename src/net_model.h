#ifndef _EMNIST_MODEL_H_
#define _EMNIST_MODEL_H_

#include <stdint.h>
#include "mli_types.h"
//=============================================
//
// Model interface
//
//=============================================
// Input tensor. To be filled with input image by user befor calling inference function (cifar10_cf_net).
#define IN_POINTS (6 * 50 * 1)
extern mli_tensor * const emnist_cf_net_input;

// Output tensor for model. Will be filled with probabilities vector by model
#define OUT_POINTS (2)
extern mli_tensor * const emnist_cf_net_output;

extern char const letters[OUT_POINTS]; 

// Model inference function
//
// Get input data from cifar10_cf_net_input tensor (FX format), fed it to the neural network,
// and writes results to cifar10_cf_net_output tensor (FX format). It is user responsibility
// to prepare input tensor correctly before calling this function and get result from output tensor
// after function finish
//
// params:
// debug_ir_root -  Path to intermediate vectors prepared in IDX format (hardcoded names). 
//                  Provides opportunity to analyse intermediate results in terms of 
//                  similarity with reference data. If path is incorrect it outputs only profiling data
//                  If NULL is passed, no messages will be printed in inference
void emnist_cf_net();

void top_n_pred(int8_t n, char *top_letters, float *top_letters_probs);
void all_pred(float *pred_data);
typedef int8_t d_type;
#endif  //_EMNIST_MODEL_H_