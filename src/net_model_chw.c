#include "emnist_model.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "mli_api.h"
#include "mli_types.h"
#include "mli_config.h"

#include "emnist_constants.h"
#define D_EL_TYPE (MLI_EL_FX_8)
//==============================================================
//
//
// Data related to the Module
//
//
//==============================================================

const char debug_ir_root[] = "model/idx";

// Intermediate data buffers (enough size for max intermediate results)
//==============================
#define IR_BUF_SZ_MOST (6*3*32)
#define IR_BUF_SZ_NEXT (6*50*2)
// #define IR_BUF_SZ_MOST (0    1
// Name 3, dtype: int64*0    6
// Name: 1, dtype: int64*0    50
// Name: 2, dtype: int64)
// #define IR_BUF_SZ_NEXT (1*0    6
// Name: 1, dtype: int64*0    50
// Name: 2, dtype: int64)

#pragma Data(".nn_ir_data_1")
static d_type  x_mem_buf[IR_BUF_SZ_MOST];
#pragma Data()

#pragma Data(".nn_ir_data_2")
static d_type  y_mem_buf[IR_BUF_SZ_NEXT];
#pragma Data()
// Module Input/Output tensors and their's external interface
//============================================================
static mli_tensor input = {
    .data = (void *)x_mem_buf,
    .capacity = sizeof(d_type) * IN_POINTS,
    .shape = {6,50,1},
    .rank = 3,
    .el_type = D_EL_TYPE,
    .el_params.fx.frac_bits = 7,
};
static mli_tensor output = {
    .data = (void *)y_mem_buf,
    .capacity = sizeof(d_type) * OUT_POINTS,
    .shape = {6,50,2},
    .rank = 3,
    .el_type = D_EL_TYPE,
    .el_params.fx.frac_bits = 0, 
};

// Interface variables: Available to user via main model header
//===========================================================
mli_tensor * const emnist_cf_net_input = &input;
mli_tensor * const emnist_cf_net_output = &output;

// char const letterss[26] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
//                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};


//==============================================================
//  Model description and configuration
//==============================================================
#pragma Data(".mli_data")

// Configuration objects for layers
//===============================================

static const mli_permute_cfg permute_hwc2chw_cfg = {
        .perm_dim = {2, 0, 1} // 2 0 1
};

static const mli_conv2d_cfg shared_conv_cfg = {
    .stride_height = 1, .stride_width = 1,
    .padding_bottom = 1, .padding_top = 1,
    .padding_left = 1, .padding_right = 1,
    .relu.type = MLI_RELU_GEN
};

static const mli_pool_cfg shared_pool_cfg = {
    .kernel_height = 1, .kernel_width = 1,
    .stride_height = 1, .stride_width = 1,
    .padding_bottom = 0, .padding_top = 0,
    .padding_left = 0, .padding_right = 0
};

// Conv 1 Layer related tensors
//===================================
static const mli_tensor L1_conv_wt = {
    .data = (void *)L1_conv_wt_buf,
    .capacity = CONV1_W_ELEMENTS * sizeof(w_type),
    .shape = CONV1_W_SHAPE,
    .rank = CONV1_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV1_W_FRAQ,
};

static const mli_tensor L1_conv_bias = {
    .data = (void *)L1_conv_bias_buf,
    .capacity = CONV1_B_ELEMENTS * sizeof(w_type),
    .shape = CONV1_B_SHAPE,
    .rank = CONV1_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV1_B_FRAQ,
};
// Conv 2 Layer related tensors
//===================================
static const mli_tensor L2_conv_wt = {
    .data = (void *)L2_conv_wt_buf,
    .capacity = CONV2_W_ELEMENTS * sizeof(w_type),
    .shape = CONV2_W_SHAPE,
    .rank = CONV2_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV2_W_FRAQ,
};

static const mli_tensor L2_conv_bias = {
    .data = (void *)L2_conv_bias_buf,
    .capacity = CONV2_B_ELEMENTS * sizeof(w_type),
    .shape = CONV2_B_SHAPE,
    .rank = CONV2_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV2_B_FRAQ,
};
// Conv 3 Layer related tensors
//===================================
static const mli_tensor L3_conv_wt = {
    .data = (void *)L3_conv_wt_buf,
    .capacity = CONV3_W_ELEMENTS * sizeof(w_type),
    .shape = CONV3_W_SHAPE,
    .rank = CONV3_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV3_W_FRAQ,
};

static const mli_tensor L3_conv_bias = {
    .data = (void *)L3_conv_bias_buf,
    .capacity = CONV3_B_ELEMENTS * sizeof(w_type),
    .shape = CONV3_B_SHAPE,
    .rank = CONV3_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV3_B_FRAQ,
};
// Conv 4 Layer related tensors
//===================================
static const mli_tensor L4_conv_wt = {
    .data = (void *)L4_conv_wt_buf,
    .capacity = CONV4_W_ELEMENTS * sizeof(w_type),
    .shape = CONV4_W_SHAPE,
    .rank = CONV4_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV4_W_FRAQ,
};

static const mli_tensor L4_conv_bias = {
    .data = (void *)L4_conv_bias_buf,
    .capacity = CONV4_B_ELEMENTS * sizeof(w_type),
    .shape = CONV4_B_SHAPE,
    .rank = CONV4_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV4_B_FRAQ,
};
// Conv 5 Layer related tensors
//===================================
static const mli_tensor L5_conv_wt = {
    .data = (void *)L5_conv_wt_buf,
    .capacity = CONV5_W_ELEMENTS * sizeof(w_type),
    .shape = CONV5_W_SHAPE,
    .rank = CONV5_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV5_W_FRAQ,
};

static const mli_tensor L5_conv_bias = {
    .data = (void *)L5_conv_bias_buf,
    .capacity = CONV5_B_ELEMENTS * sizeof(w_type),
    .shape = CONV5_B_SHAPE,
    .rank = CONV5_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV5_B_FRAQ,
};
// Conv 6 Layer related tensors
//===================================
static const mli_tensor L6_conv_wt = {
    .data = (void *)L6_conv_wt_buf,
    .capacity = CONV6_W_ELEMENTS * sizeof(w_type),
    .shape = CONV6_W_SHAPE,
    .rank = CONV6_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV6_W_FRAQ,
};

static const mli_tensor L6_conv_bias = {
    .data = (void *)L6_conv_bias_buf,
    .capacity = CONV6_B_ELEMENTS * sizeof(w_type),
    .shape = CONV6_B_SHAPE,
    .rank = CONV6_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV6_B_FRAQ,
};
// Conv 7 Layer related tensors
//===================================
static const mli_tensor L7_conv_wt = {
    .data = (void *)L7_conv_wt_buf,
    .capacity = CONV7_W_ELEMENTS * sizeof(w_type),
    .shape = CONV7_W_SHAPE,
    .rank = CONV7_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV7_W_FRAQ,
};

static const mli_tensor L7_conv_bias = {
    .data = (void *)L7_conv_bias_buf,
    .capacity = CONV7_B_ELEMENTS * sizeof(w_type),
    .shape = CONV7_B_SHAPE,
    .rank = CONV7_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV7_B_FRAQ,
};
// Conv 8 Layer related tensors
//===================================
static const mli_tensor L8_conv_wt = {
    .data = (void *)L8_conv_wt_buf,
    .capacity = CONV8_W_ELEMENTS * sizeof(w_type),
    .shape = CONV8_W_SHAPE,
    .rank = CONV8_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV8_W_FRAQ,
};

static const mli_tensor L8_conv_bias = {
    .data = (void *)L8_conv_bias_buf,
    .capacity = CONV8_B_ELEMENTS * sizeof(w_type),
    .shape = CONV8_B_SHAPE,
    .rank = CONV8_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV8_B_FRAQ,
};

// Conv 9 Layer related tensors
//===================================
static const mli_tensor L9_conv_wt = {
    .data = (void *)L9_conv_wt_buf,
    .capacity = CONV9_W_ELEMENTS * sizeof(w_type),
    .shape = CONV9_W_SHAPE,
    .rank = CONV9_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV9_W_FRAQ,
};

static const mli_tensor L9_conv_bias = {
    .data = (void *)L9_conv_bias_buf,
    .capacity = CONV9_B_ELEMENTS * sizeof(w_type),
    .shape = CONV9_B_SHAPE,
    .rank = CONV9_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV9_B_FRAQ,
};
// Conv 10 Layer related tensors
//===================================
static const mli_tensor L10_conv_wt = {
    .data = (void *)L10_conv_wt_buf,
    .capacity = CONV10_W_ELEMENTS * sizeof(w_type),
    .shape = CONV10_W_SHAPE,
    .rank = CONV10_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV10_W_FRAQ,
};

static const mli_tensor L10_conv_bias = {
    .data = (void *)L10_conv_bias_buf,
    .capacity = CONV10_B_ELEMENTS * sizeof(w_type),
    .shape = CONV10_B_SHAPE,
    .rank = CONV10_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV10_B_FRAQ,
};

// Conv 11 Layer related tensors
//===================================
static const mli_tensor L11_conv_wt = {
    .data = (void *)L11_conv_wt_buf,
    .capacity = CONV11_W_ELEMENTS * sizeof(w_type),
    .shape = CONV11_W_SHAPE,
    .rank = CONV11_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV11_W_FRAQ,
};

static const mli_tensor L11_conv_bias = {
    .data = (void *)L11_conv_bias_buf,
    .capacity = CONV11_B_ELEMENTS * sizeof(w_type),
    .shape = CONV11_B_SHAPE,
    .rank = CONV11_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV11_B_FRAQ,
};
// Conv 12 Layer related tensors
//===================================
static const mli_tensor L12_conv_wt = {
    .data = (void *)L12_conv_wt_buf,
    .capacity = CONV12_W_ELEMENTS * sizeof(w_type),
    .shape = CONV12_W_SHAPE,
    .rank = CONV12_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV12_W_FRAQ,
};

static const mli_tensor L12_conv_bias = {
    .data = (void *)L12_conv_bias_buf,
    .capacity = CONV12_B_ELEMENTS * sizeof(w_type),
    .shape = CONV12_B_SHAPE,
    .rank = CONV12_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV12_B_FRAQ,
};
// Conv 13 Layer related tensors
//===================================
static const mli_tensor L13_conv_wt = {
    .data = (void *)L13_conv_wt_buf,
    .capacity = CONV13_W_ELEMENTS * sizeof(w_type),
    .shape = CONV13_W_SHAPE,
    .rank = CONV13_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV13_W_FRAQ,
};

static const mli_tensor L13_conv_bias = {
    .data = (void *)L13_conv_bias_buf,
    .capacity = CONV13_B_ELEMENTS * sizeof(w_type),
    .shape = CONV13_B_SHAPE,
    .rank = CONV13_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV13_B_FRAQ,
};
// Conv 14 Layer related tensors
//===================================
static const mli_tensor L14_conv_wt = {
    .data = (void *)L14_conv_wt_buf,
    .capacity = CONV14_W_ELEMENTS * sizeof(w_type),
    .shape = CONV14_W_SHAPE,
    .rank = CONV14_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV14_W_FRAQ,
};

static const mli_tensor L14_conv_bias = {
    .data = (void *)L14_conv_bias_buf,
    .capacity = CONV14_B_ELEMENTS * sizeof(w_type),
    .shape = CONV14_B_SHAPE,
    .rank = CONV14_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV14_B_FRAQ,
};
// Conv 15 Layer related tensors
//===================================
static const mli_tensor L15_conv_wt = {
    .data = (void *)L15_conv_wt_buf,
    .capacity = CONV15_W_ELEMENTS * sizeof(w_type),
    .shape = CONV15_W_SHAPE,
    .rank = CONV15_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV15_W_FRAQ,
};

static const mli_tensor L15_conv_bias = {
    .data = (void *)L15_conv_bias_buf,
    .capacity = CONV15_B_ELEMENTS * sizeof(w_type),
    .shape = CONV15_B_SHAPE,
    .rank = CONV15_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV15_B_FRAQ,
};
// Conv 16 Layer related tensors
//===================================
static const mli_tensor L16_conv_wt = {
    .data = (void *)L16_conv_wt_buf,
    .capacity = CONV16_W_ELEMENTS * sizeof(w_type),
    .shape = CONV16_W_SHAPE,
    .rank = CONV16_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV16_W_FRAQ,
};

static const mli_tensor L16_conv_bias = {
    .data = (void *)L16_conv_bias_buf,
    .capacity = CONV16_B_ELEMENTS * sizeof(w_type),
    .shape = CONV16_B_SHAPE,
    .rank = CONV16_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV16_B_FRAQ,
};
// Conv 17 Layer related tensors
//===================================
static const mli_tensor L17_conv_wt = {
    .data = (void *)L17_conv_wt_buf,
    .capacity = CONV17_W_ELEMENTS * sizeof(w_type),
    .shape = CONV17_W_SHAPE,
    .rank = CONV17_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV17_W_FRAQ,
};

static const mli_tensor L17_conv_bias = {
    .data = (void *)L17_conv_bias_buf,
    .capacity = CONV17_B_ELEMENTS * sizeof(w_type),
    .shape = CONV17_B_SHAPE,
    .rank = CONV17_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV17_B_FRAQ,
};
// Conv 18 Layer related tensors
//===================================
static const mli_tensor L18_conv_wt = {
    .data = (void *)L18_conv_wt_buf,
    .capacity = CONV18_W_ELEMENTS * sizeof(w_type),
    .shape = CONV18_W_SHAPE,
    .rank = CONV18_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV18_W_FRAQ,
};

static const mli_tensor L18_conv_bias = {
    .data = (void *)L18_conv_bias_buf,
    .capacity = CONV18_B_ELEMENTS * sizeof(w_type),
    .shape = CONV18_B_SHAPE,
    .rank = CONV18_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV18_B_FRAQ,
};
// Conv 19 Layer related tensors
//===================================
static const mli_tensor L19_conv_wt = {
    .data = (void *)L19_conv_wt_buf,
    .capacity = CONV19_W_ELEMENTS * sizeof(w_type),
    .shape = CONV19_W_SHAPE,
    .rank = CONV19_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV19_W_FRAQ,
};

static const mli_tensor L19_conv_bias = {
    .data = (void *)L19_conv_bias_buf,
    .capacity = CONV19_B_ELEMENTS * sizeof(w_type),
    .shape = CONV19_B_SHAPE,
    .rank = CONV19_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV19_B_FRAQ,
};
// Conv 20 Layer related tensors
//===================================
static const mli_tensor L20_conv_wt = {
    .data = (void *)L20_conv_wt_buf,
    .capacity = CONV20_W_ELEMENTS * sizeof(w_type),
    .shape = CONV20_W_SHAPE,
    .rank = CONV20_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV20_W_FRAQ,
};

static const mli_tensor L20_conv_bias = {
    .data = (void *)L20_conv_bias_buf,
    .capacity = CONV20_B_ELEMENTS * sizeof(w_type),
    .shape = CONV20_B_SHAPE,
    .rank = CONV20_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV20_B_FRAQ,
};
// Conv 21 Layer related tensors
//===================================
static const mli_tensor L21_conv_wt = {
    .data = (void *)L21_conv_wt_buf,
    .capacity = CONV21_W_ELEMENTS * sizeof(w_type),
    .shape = CONV21_W_SHAPE,
    .rank = CONV21_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV21_W_FRAQ,
};

static const mli_tensor L21_conv_bias = {
    .data = (void *)L21_conv_bias_buf,
    .capacity = CONV21_B_ELEMENTS * sizeof(w_type),
    .shape = CONV21_B_SHAPE,
    .rank = CONV21_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV21_B_FRAQ,
};
// Conv 22 Layer related tensors
//===================================
static const mli_tensor L22_conv_wt = {
    .data = (void *)L22_conv_wt_buf,
    .capacity = CONV22_W_ELEMENTS * sizeof(w_type),
    .shape = CONV22_W_SHAPE,
    .rank = CONV22_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV22_W_FRAQ,
};

static const mli_tensor L22_conv_bias = {
    .data = (void *)L22_conv_bias_buf,
    .capacity = CONV22_B_ELEMENTS * sizeof(w_type),
    .shape = CONV22_B_SHAPE,
    .rank = CONV22_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV22_B_FRAQ,
};
// Conv 23 Layer related tensors
//===================================
static const mli_tensor L23_conv_wt = {
    .data = (void *)L23_conv_wt_buf,
    .capacity = CONV23_W_ELEMENTS * sizeof(w_type),
    .shape = CONV23_W_SHAPE,
    .rank = CONV23_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV23_W_FRAQ,
};

static const mli_tensor L23_conv_bias = {
    .data = (void *)L23_conv_bias_buf,
    .capacity = CONV23_B_ELEMENTS * sizeof(w_type),
    .shape = CONV23_B_SHAPE,
    .rank = CONV23_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV23_B_FRAQ,
};
// Conv 24 Layer related tensors
//===================================
static const mli_tensor L24_conv_wt = {
    .data = (void *)L24_conv_wt_buf,
    .capacity = CONV24_W_ELEMENTS * sizeof(w_type),
    .shape = CONV24_W_SHAPE,
    .rank = CONV24_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV24_W_FRAQ,
};

static const mli_tensor L24_conv_bias = {
    .data = (void *)L24_conv_bias_buf,
    .capacity = CONV24_B_ELEMENTS * sizeof(w_type),
    .shape = CONV24_B_SHAPE,
    .rank = CONV24_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV24_B_FRAQ,
};
// Conv 25 Layer related tensors
//===================================
static const mli_tensor L25_conv_wt = {
    .data = (void *)L25_conv_wt_buf,
    .capacity = CONV25_W_ELEMENTS * sizeof(w_type),
    .shape = CONV25_W_SHAPE,
    .rank = CONV25_W_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV25_W_FRAQ,
};

static const mli_tensor L25_conv_bias = {
    .data = (void *)L25_conv_bias_buf,
    .capacity = CONV25_B_ELEMENTS * sizeof(w_type),
    .shape = CONV25_B_SHAPE,
    .rank = CONV25_B_RANK,
    .el_type = W_EL_TYPE,
    .el_params.fx.frac_bits = CONV25_B_FRAQ,
};
// Intermediate result tensors
//===============================================
static mli_tensor ir_tensor_X = {
    .data = (void *)x_mem_buf,
    .capacity = sizeof(x_mem_buf),
    .shape = {0, 0, 0, 0},
    .rank = 4,
    .el_type = D_EL_TYPE,
    .el_params.fx.frac_bits = FRQ_BITS(0, d_type),
};

static mli_tensor ir_tensor_Y = {
    .data = (void *)y_mem_buf,
    .capacity = sizeof(y_mem_buf),
    .shape = {0, 0, 0, 0},
    .rank = 4,
    .el_type = D_EL_TYPE,
    .el_params.fx.frac_bits = FRQ_BITS(0, d_type),
};

#pragma Data()

static void preprocessing(mli_tensor* net_input_) {
    d_type* const dst = (d_type * const)net_input_->data;
    // Acceleration preprocessing
    if (net_input_->el_params.fx.frac_bits == 7) {
        for (int idx = 0; idx < IN_POINTS/2; idx++) {
            dst[idx] = (int)(dst[idx]/16.0);
        }
    }
    else if (net_input_->el_params.fx.frac_bits > 7) {
        int shift_left = net_input_->el_params.fx.frac_bits - 7;
        for (int idx = 0; idx < IN_POINTS/2; idx++) {
            dst[idx] = (int)(dst[idx]/16.0) << shift_left;
        }
    }
    else {
        int shift_right = 7 - net_input_->el_params.fx.frac_bits;
        for (int idx = 0; idx < IN_POINTS/2; idx++) {
            dst[idx] = (int)(dst[idx]/16.0) >> shift_right;
        }
    }
    // Gyro preprocessing
    if (net_input_->el_params.fx.frac_bits == 7) {
        for (int idx = IN_POINTS/2; idx < IN_POINTS; idx++) {
            dst[idx] = (dst[idx]/2000.0);
        }
    }
    else if (net_input_->el_params.fx.frac_bits > 7) {
        int shift_left = net_input_->el_params.fx.frac_bits - 7;
        for (int idx = IN_POINTS/2; idx < IN_POINTS; idx++) {
            dst[idx] = (int)((dst[idx])/2000.0 )<< shift_left;
        }
    }
    else {
        int shift_right = 7 - net_input_->el_params.fx.frac_bits;
        for (int idx = IN_POINTS/2; idx < IN_POINTS; idx++) {
            dst[idx] = (int)(dst[idx]/2000.0) >> shift_right;
        }
    }
}
static void tensor_to_float (const mli_tensor * src, float *dst, uint32_t dst_size) {
    const float scale_val = 1.0f / (float) (1u << (src->el_params.fx.frac_bits));
    if (src->el_type == MLI_EL_FX_16) {
        int16_t *src_arr = src->data;
        for (int idx = 0; idx < dst_size; idx++)
            dst[idx] = (float) (scale_val * src_arr[idx]);
    } else {
        int8_t *src_arr = src->data;
        for (int idx = 0; idx < dst_size; idx++)
            dst[idx] = (float) (scale_val * src_arr[idx]);
    }
}


// void top_n_pred(int8_t n, char *top_letters, float *top_letters_probs) {
//     uint8_t flags[OUT_POINTS] = {0};
//     float pred_data[OUT_POINTS] = {0};
//     //d_type* const out = (d_type * const)emnist_cf_net_output->data;
//     tensor_to_float(emnist_cf_net_output, pred_data, OUT_POINTS);
//     for (int top = 0; top < n; top++) {
//         float max = -1;
//         uint8_t max_idx = -1;
        
//         for (int idx = 0; idx < OUT_POINTS; idx++) {
//             if(pred_data[idx] > max && flags[idx] != 1) {
//                 max = pred_data[idx];
//                 max_idx = idx;
//             }
//         }

//         top_letters[top] = letterss[max_idx];
//         top_letters_probs[top] = pred_data[max_idx];
//         flags[max_idx] = 1;
//     }
// }


void all_pred(float *pred_data) {
    tensor_to_float(emnist_cf_net_output, pred_data, OUT_POINTS);
}
static inline mli_status softmax(const mli_tensor *in,  mli_tensor *out) {
    return mli_krn_softmax_fx8(in, out);
}

static const mli_relu_cfg relu_cfg = {.type = MLI_RELU_GEN};
static inline mli_status relu(const mli_tensor *in, const mli_relu_cfg *cfg, mli_tensor *out) {
    return mli_krn_relu_fx8(in, cfg, out);
}

static inline mli_status mli_krn_permute_fx(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {
    return mli_krn_permute_fx8(in, cfg, out);
}

static inline mli_status maxpool_chw(const mli_tensor *in, const mli_pool_cfg *cfg, mli_tensor *out) {
    /* GENERIC VERSION OF KERNEL IS USED BELOW. REPLACE IT WITH SPECIALIZED ONE FOR BETTER PERFORMANCE. */
    return mli_krn_maxpool_chw_fx8_generic(in, cfg, out);
}


static inline mli_status conv2d_chw(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor *bias,
        const mli_conv2d_cfg *cfg,
        mli_tensor *out) {
    /* GENERIC VERSION OF KERNEL IS USED BELOW. REPLACE IT WITH SPECIALIZED ONE FOR BETTER PERFORMANCE. */
    return mli_krn_conv2d_chw_fx8_generic(in, weights, bias, cfg, out);
}

static inline mli_status fully_connected(
        const mli_tensor *in,
        const mli_tensor *weights,
        const mli_tensor *bias,
        mli_tensor *out) {
    return mli_krn_fully_connected_fx8(in, weights, bias, out);
}

//==============================================================
//
//  EMNIST graph based on Keras example.
//  Layer-by-Layer execution for CHW layput
//
//==============================================================
void emnist_cf_net() {
        preprocessing(&input);
    

        mli_krn_permute_fx(&input, &permute_hwc2chw_cfg, &ir_tensor_Y);
        //conv1
        ir_tensor_X.el_params.fx.frac_bits = CONV1_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L1_conv_wt, &L1_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    
        //conv2
        ir_tensor_Y.el_params.fx.frac_bits = CONV2_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L2_conv_wt, &L2_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    

        maxpool_chw(&ir_tensor_Y, &shared_pool_cfg, &ir_tensor_X);
    
        //conv3
        ir_tensor_Y.el_params.fx.frac_bits = CONV3_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L3_conv_wt, &L3_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    
        //conv4
        ir_tensor_X.el_params.fx.frac_bits = CONV4_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L4_conv_wt, &L4_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    

        maxpool_chw(&ir_tensor_X, &shared_pool_cfg, &ir_tensor_Y);
    
        //conv5
        ir_tensor_X.el_params.fx.frac_bits = CONV5_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L5_conv_wt, &L5_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    
        //conv6
        ir_tensor_Y.el_params.fx.frac_bits = CONV6_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L6_conv_wt, &L6_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    

        maxpool_chw(&ir_tensor_Y, &shared_pool_cfg, &ir_tensor_X);
    
        //conv7
        ir_tensor_Y.el_params.fx.frac_bits = CONV7_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L7_conv_wt, &L7_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    
        //conv8
        ir_tensor_X.el_params.fx.frac_bits = CONV8_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L8_conv_wt, &L8_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    

        maxpool_chw(&ir_tensor_X, &shared_pool_cfg, &ir_tensor_Y);
    
        //conv9
        ir_tensor_X.el_params.fx.frac_bits = CONV9_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L9_conv_wt, &L9_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    
        //conv10
        ir_tensor_Y.el_params.fx.frac_bits = CONV10_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L10_conv_wt, &L10_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    
        //conv11
        ir_tensor_X.el_params.fx.frac_bits = CONV11_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L11_conv_wt, &L11_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    
        //conv12
        ir_tensor_Y.el_params.fx.frac_bits = CONV12_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L12_conv_wt, &L12_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    
        //conv13
        ir_tensor_X.el_params.fx.frac_bits = CONV13_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L13_conv_wt, &L13_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    
        //conv14
        ir_tensor_Y.el_params.fx.frac_bits = CONV14_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L14_conv_wt, &L14_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    
        //conv15
        ir_tensor_X.el_params.fx.frac_bits = CONV15_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L15_conv_wt, &L15_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    
        //conv16
        ir_tensor_Y.el_params.fx.frac_bits = CONV16_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L16_conv_wt, &L16_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    
        //conv17
        ir_tensor_X.el_params.fx.frac_bits = CONV17_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L17_conv_wt, &L17_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    
        //conv18
        ir_tensor_Y.el_params.fx.frac_bits = CONV18_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L18_conv_wt, &L18_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    
        //conv19
        ir_tensor_X.el_params.fx.frac_bits = CONV19_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L19_conv_wt, &L19_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    
        //conv20
        ir_tensor_Y.el_params.fx.frac_bits = CONV20_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L20_conv_wt, &L20_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    
        //conv21
        ir_tensor_X.el_params.fx.frac_bits = CONV21_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L21_conv_wt, &L21_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    
        //conv22
        ir_tensor_Y.el_params.fx.frac_bits = CONV22_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L22_conv_wt, &L22_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    
        //conv23
        ir_tensor_X.el_params.fx.frac_bits = CONV23_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L23_conv_wt, &L23_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    
        //conv24
        ir_tensor_Y.el_params.fx.frac_bits = CONV24_OUT_FRAQ;
        conv2d_chw(&ir_tensor_X, &L24_conv_wt, &L24_conv_bias, &shared_conv_cfg, &ir_tensor_Y);
    
        //conv25
        ir_tensor_X.el_params.fx.frac_bits = CONV25_OUT_FRAQ;
        conv2d_chw(&ir_tensor_Y, &L25_conv_wt, &L25_conv_bias, &shared_conv_cfg, &ir_tensor_X);
    
}


