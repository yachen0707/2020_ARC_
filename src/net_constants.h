#ifndef _EMNIST_CONSTANTS_H_
#define _EMNIST_CONSTANTS_H_
#include "mli_config.h"
#include "emnist_model.h"
#define W_EL_TYPE (MLI_EL_FX_8)
typedef int8_t w_type;
// Defining data sections attributes
//===================================
#if (ARC_PLATFORM == V2DSP_XY)
#if defined (__GNUC__) && !defined (__CCAC__)
// ARC GNU tools
// Model Weights attribute
#define _Wdata_attr __attribute__((section(".mli_model")))
#define _W  _Wdata_attr

// Model Weights (part 2) attribute
#define _W2data_attr __attribute__((section(".mli_model_p2")))
#define _W2  _W2data_attr

// Bank X (XCCM) attribute
#define __Xdata_attr __attribute__((section(".Xdata")))
#define _X  __Xdata_attr

// Bank Y (YCCM) attribute
#define __Ydata_attr __attribute__((section(".Ydata")))
#define _Y  __Ydata_attr

// Bank Z (DCCM) attribute
#define __Zdata_attr __attribute__((section(".Zdata")))
#define _Z  __Zdata_attr

#else
// Metaware tools
// Model Weights attribute
#define _Wdata_attr __attribute__((section(".mli_model")))
#define _W __xy _Wdata_attr

// Model Weights (part 2) attribute
#define _W2data_attr __attribute__((section(".mli_model_p2")))
#define _W2 __xy _W2data_attr

// Bank X (XCCM) attribute
#define __Xdata_attr __attribute__((section(".Xdata")))
#define _X __xy __Xdata_attr

// Bank Y (YCCM) attribute
#define __Ydata_attr __attribute__((section(".Ydata")))
#define _Y __xy __Ydata_attr

// Bank Z (DCCM) attribute
#define __Zdata_attr __attribute__((section(".Zdata")))
#define _Z __xy __Zdata_attr
#endif // if defined (__GNUC__) && !defined (__CCAC__)

#else
#define _X __attribute__((section(".mli_ir_buf")))
#define _Y __attribute__((section(".mli_ir_buf")))
#define _Z __attribute__((section(".mli_ir_buf")))
#define _W __attribute__((section(".mli_model")))
#define _W2 __attribute__((section(".mli_model")))
#endif

//======================================================
//
// Common data transform (Qmn) defines
//
//======================================================

#define QMN(type, fraq, val)   (type)(val * (1u << (fraq)) + ((val >= 0)? 0.5f: -0.5f))
#define FRQ_BITS(int_part, el_type) ((sizeof(el_type)*8)-int_part-1)

//======================================================
//
// Common data transform (Qmn) defines
//
//======================================================
extern const w_type _W L1_conv_wt_buf[];
extern const w_type _W L1_conv_bias_buf[];
extern const w_type _W L2_conv_wt_buf[];
extern const w_type _W L2_conv_bias_buf[];
extern const w_type _W L3_conv_wt_buf[];
extern const w_type _W L3_conv_bias_buf[];
extern const w_type _W L4_conv_wt_buf[];
extern const w_type _W L4_conv_bias_buf[];
extern const w_type _W L5_conv_wt_buf[];
extern const w_type _W L5_conv_bias_buf[];
extern const w_type _W L6_conv_wt_buf[];
extern const w_type _W L6_conv_bias_buf[];
extern const w_type _W L7_conv_wt_buf[];
extern const w_type _W L7_conv_bias_buf[];
extern const w_type _W L8_conv_wt_buf[];
extern const w_type _W L8_conv_bias_buf[];
extern const w_type _W L9_conv_wt_buf[];
extern const w_type _W L9_conv_bias_buf[];
extern const w_type _W L10_conv_wt_buf[];
extern const w_type _W L10_conv_bias_buf[];
extern const w_type _W L11_conv_wt_buf[];
extern const w_type _W L11_conv_bias_buf[];
extern const w_type _W L12_conv_wt_buf[];
extern const w_type _W L12_conv_bias_buf[];
extern const w_type _W2 L13_conv_wt_buf[];
extern const w_type _W2 L13_conv_bias_buf[];
extern const w_type _W2 L14_conv_wt_buf[];
extern const w_type _W2 L14_conv_bias_buf[];
extern const w_type _W2 L15_conv_wt_buf[];
extern const w_type _W2 L15_conv_bias_buf[];
extern const w_type _W2 L16_conv_wt_buf[];
extern const w_type _W2 L16_conv_bias_buf[];
extern const w_type _W2 L17_conv_wt_buf[];
extern const w_type _W2 L17_conv_bias_buf[];
extern const w_type _W2 L18_conv_wt_buf[];
extern const w_type _W2 L18_conv_bias_buf[];
extern const w_type _W2 L19_conv_wt_buf[];
extern const w_type _W2 L19_conv_bias_buf[];
extern const w_type _W2 L20_conv_wt_buf[];
extern const w_type _W2 L20_conv_bias_buf[];
extern const w_type _W2 L21_conv_wt_buf[];
extern const w_type _W2 L21_conv_bias_buf[];
extern const w_type _W2 L22_conv_wt_buf[];
extern const w_type _W2 L22_conv_bias_buf[];
extern const w_type _W2 L23_conv_wt_buf[];
extern const w_type _W2 L23_conv_bias_buf[];
extern const w_type _W2 L24_conv_wt_buf[];
extern const w_type _W2 L24_conv_bias_buf[];
extern const w_type _W2 L25_conv_wt_buf[];
extern const w_type _W2 L25_conv_bias_buf[];




#define CONV1_W_INT   (1)
#define CONV1_B_INT   (-3)
#define CONV1_OUT_INT (0)

#define CONV2_W_INT   (0)
#define CONV2_B_INT   (-3)
#define CONV2_OUT_INT (128)



#define CONV3_W_INT   (0)
#define CONV3_B_INT   (-2)
#define CONV3_OUT_INT (128)

#define CONV4_W_INT   (0)
#define CONV4_B_INT   (-3)
#define CONV4_OUT_INT (128)



#define CONV5_W_INT   (0)
#define CONV5_B_INT   (-2)
#define CONV5_OUT_INT (128)

#define CONV6_W_INT   (0)
#define CONV6_B_INT   (-2)
#define CONV6_OUT_INT (128)



#define CONV7_W_INT   (0)
#define CONV7_B_INT   (-2)
#define CONV7_OUT_INT (128)

#define CONV8_W_INT   (-1)
#define CONV8_B_INT   (-3)
#define CONV8_OUT_INT (128)





#define CONV9_W_INT   (0)
#define CONV9_B_INT   (-2)
#define CONV9_OUT_INT (128)

#define CONV10_W_INT   (-1)
#define CONV10_B_INT   (-3)
#define CONV10_OUT_INT (128)





#define CONV11_W_INT   (-1)
#define CONV11_B_INT   (-2)
#define CONV11_OUT_INT (128)



#define CONV12_W_INT   (0)
#define CONV12_B_INT   (-2)
#define CONV12_OUT_INT (128)

#define CONV13_W_INT   (0)
#define CONV13_B_INT   (-2)
#define CONV13_OUT_INT (128)



#define CONV14_W_INT   (0)
#define CONV14_B_INT   (-2)
#define CONV14_OUT_INT (128)



#define CONV15_W_INT   (0)
#define CONV15_B_INT   (-2)
#define CONV15_OUT_INT (128)

#define CONV16_W_INT   (0)
#define CONV16_B_INT   (-1)
#define CONV16_OUT_INT (128)

#define CONV17_W_INT   (1)
#define CONV17_B_INT   (-1)
#define CONV17_OUT_INT (128)

#define CONV18_W_INT   (1)
#define CONV18_B_INT   (-2)
#define CONV18_OUT_INT (128)



#define CONV19_W_INT   (1)
#define CONV19_B_INT   (-1)
#define CONV19_OUT_INT (128)

#define CONV20_W_INT   (1)
#define CONV20_B_INT   (-1)
#define CONV20_OUT_INT (128)



#define CONV21_W_INT   (1)
#define CONV21_B_INT   (1)
#define CONV21_OUT_INT (128)



#define CONV22_W_INT   (1)
#define CONV22_B_INT   (-1)
#define CONV22_OUT_INT (128)

#define CONV23_W_INT   (1)
#define CONV23_B_INT   (0)
#define CONV23_OUT_INT (128)

#define CONV24_W_INT   (1)
#define CONV24_B_INT   (0)
#define CONV24_OUT_INT (128)

#define CONV25_W_INT   (1)
#define CONV25_B_INT   (2)
#define CONV25_OUT_INT (128)



// CONV1
//================================================
#define CONV1_W_SHAPE {2,1,3,3}
#define CONV1_W_ELEMENTS (2*1*3*3)
#define CONV1_W_RANK (4)

#define CONV1_W_FRAQ   (FRQ_BITS(CONV1_W_INT, w_type))
#define L1_WQ(val)   QMN(w_type, CONV1_W_FRAQ, val)

#define CONV1_B_ELEMENTS (2)
#define CONV1_B_SHAPE {2}
#define CONV1_B_RANK (1)

#define CONV1_B_FRAQ   (FRQ_BITS(CONV1_B_INT, w_type))
#define L1_BQ(val)   QMN(w_type, CONV1_B_FRAQ, val)

#define CONV1_OUT_FRAQ (FRQ_BITS(CONV1_OUT_INT, d_type))

// CONV2
//================================================
#define CONV2_W_SHAPE {2,2,3,3}
#define CONV2_W_ELEMENTS (2*2*3*3)
#define CONV2_W_RANK (4)

#define CONV2_W_FRAQ   (FRQ_BITS(CONV2_W_INT, w_type))
#define L2_WQ(val)   QMN(w_type, CONV2_W_FRAQ, val)

#define CONV2_B_ELEMENTS (2)
#define CONV2_B_SHAPE {2}
#define CONV2_B_RANK (1)

#define CONV2_B_FRAQ   (FRQ_BITS(CONV2_B_INT, w_type))
#define L2_BQ(val)   QMN(w_type, CONV2_B_FRAQ, val)

#define CONV2_OUT_FRAQ (FRQ_BITS(CONV2_OUT_INT, d_type))



// CONV3
//================================================
#define CONV3_W_SHAPE {4,2,3,3}
#define CONV3_W_ELEMENTS (4*2*3*3)
#define CONV3_W_RANK (4)

#define CONV3_W_FRAQ   (FRQ_BITS(CONV3_W_INT, w_type))
#define L3_WQ(val)   QMN(w_type, CONV3_W_FRAQ, val)

#define CONV3_B_ELEMENTS (4)
#define CONV3_B_SHAPE {4}
#define CONV3_B_RANK (1)

#define CONV3_B_FRAQ   (FRQ_BITS(CONV3_B_INT, w_type))
#define L3_BQ(val)   QMN(w_type, CONV3_B_FRAQ, val)

#define CONV3_OUT_FRAQ (FRQ_BITS(CONV3_OUT_INT, d_type))

// CONV4
//================================================
#define CONV4_W_SHAPE {4,4,3,3}
#define CONV4_W_ELEMENTS (4*4*3*3)
#define CONV4_W_RANK (4)

#define CONV4_W_FRAQ   (FRQ_BITS(CONV4_W_INT, w_type))
#define L4_WQ(val)   QMN(w_type, CONV4_W_FRAQ, val)

#define CONV4_B_ELEMENTS (4)
#define CONV4_B_SHAPE {4}
#define CONV4_B_RANK (1)

#define CONV4_B_FRAQ   (FRQ_BITS(CONV4_B_INT, w_type))
#define L4_BQ(val)   QMN(w_type, CONV4_B_FRAQ, val)

#define CONV4_OUT_FRAQ (FRQ_BITS(CONV4_OUT_INT, d_type))



// CONV5
//================================================
#define CONV5_W_SHAPE {8,4,3,3}
#define CONV5_W_ELEMENTS (8*4*3*3)
#define CONV5_W_RANK (4)

#define CONV5_W_FRAQ   (FRQ_BITS(CONV5_W_INT, w_type))
#define L5_WQ(val)   QMN(w_type, CONV5_W_FRAQ, val)

#define CONV5_B_ELEMENTS (8)
#define CONV5_B_SHAPE {8}
#define CONV5_B_RANK (1)

#define CONV5_B_FRAQ   (FRQ_BITS(CONV5_B_INT, w_type))
#define L5_BQ(val)   QMN(w_type, CONV5_B_FRAQ, val)

#define CONV5_OUT_FRAQ (FRQ_BITS(CONV5_OUT_INT, d_type))

// CONV6
//================================================
#define CONV6_W_SHAPE {8,8,3,3}
#define CONV6_W_ELEMENTS (8*8*3*3)
#define CONV6_W_RANK (4)

#define CONV6_W_FRAQ   (FRQ_BITS(CONV6_W_INT, w_type))
#define L6_WQ(val)   QMN(w_type, CONV6_W_FRAQ, val)

#define CONV6_B_ELEMENTS (8)
#define CONV6_B_SHAPE {8}
#define CONV6_B_RANK (1)

#define CONV6_B_FRAQ   (FRQ_BITS(CONV6_B_INT, w_type))
#define L6_BQ(val)   QMN(w_type, CONV6_B_FRAQ, val)

#define CONV6_OUT_FRAQ (FRQ_BITS(CONV6_OUT_INT, d_type))



// CONV7
//================================================
#define CONV7_W_SHAPE {16,8,3,3}
#define CONV7_W_ELEMENTS (16*8*3*3)
#define CONV7_W_RANK (4)

#define CONV7_W_FRAQ   (FRQ_BITS(CONV7_W_INT, w_type))
#define L7_WQ(val)   QMN(w_type, CONV7_W_FRAQ, val)

#define CONV7_B_ELEMENTS (16)
#define CONV7_B_SHAPE {16}
#define CONV7_B_RANK (1)

#define CONV7_B_FRAQ   (FRQ_BITS(CONV7_B_INT, w_type))
#define L7_BQ(val)   QMN(w_type, CONV7_B_FRAQ, val)

#define CONV7_OUT_FRAQ (FRQ_BITS(CONV7_OUT_INT, d_type))

// CONV8
//================================================
#define CONV8_W_SHAPE {16,16,3,3}
#define CONV8_W_ELEMENTS (16*16*3*3)
#define CONV8_W_RANK (4)

#define CONV8_W_FRAQ   (FRQ_BITS(CONV8_W_INT, w_type))
#define L8_WQ(val)   QMN(w_type, CONV8_W_FRAQ, val)

#define CONV8_B_ELEMENTS (16)
#define CONV8_B_SHAPE {16}
#define CONV8_B_RANK (1)

#define CONV8_B_FRAQ   (FRQ_BITS(CONV8_B_INT, w_type))
#define L8_BQ(val)   QMN(w_type, CONV8_B_FRAQ, val)

#define CONV8_OUT_FRAQ (FRQ_BITS(CONV8_OUT_INT, d_type))





// CONV9
//================================================
#define CONV9_W_SHAPE {32,16,3,3}
#define CONV9_W_ELEMENTS (32*16*3*3)
#define CONV9_W_RANK (4)

#define CONV9_W_FRAQ   (FRQ_BITS(CONV9_W_INT, w_type))
#define L9_WQ(val)   QMN(w_type, CONV9_W_FRAQ, val)

#define CONV9_B_ELEMENTS (32)
#define CONV9_B_SHAPE {32}
#define CONV9_B_RANK (1)

#define CONV9_B_FRAQ   (FRQ_BITS(CONV9_B_INT, w_type))
#define L9_BQ(val)   QMN(w_type, CONV9_B_FRAQ, val)

#define CONV9_OUT_FRAQ (FRQ_BITS(CONV9_OUT_INT, d_type))

// CONV10
//================================================
#define CONV10_W_SHAPE {32,32,3,3}
#define CONV10_W_ELEMENTS (32*32*3*3)
#define CONV10_W_RANK (4)

#define CONV10_W_FRAQ   (FRQ_BITS(CONV10_W_INT, w_type))
#define L10_WQ(val)   QMN(w_type, CONV10_W_FRAQ, val)

#define CONV10_B_ELEMENTS (32)
#define CONV10_B_SHAPE {32}
#define CONV10_B_RANK (1)

#define CONV10_B_FRAQ   (FRQ_BITS(CONV10_B_INT, w_type))
#define L10_BQ(val)   QMN(w_type, CONV10_B_FRAQ, val)

#define CONV10_OUT_FRAQ (FRQ_BITS(CONV10_OUT_INT, d_type))





// CONV11
//================================================
#define CONV11_W_SHAPE {16,32,1,2}
#define CONV11_W_ELEMENTS (16*32*1*2)
#define CONV11_W_RANK (4)

#define CONV11_W_FRAQ   (FRQ_BITS(CONV11_W_INT, w_type))
#define L11_WQ(val)   QMN(w_type, CONV11_W_FRAQ, val)

#define CONV11_B_ELEMENTS (16)
#define CONV11_B_SHAPE {16}
#define CONV11_B_RANK (1)

#define CONV11_B_FRAQ   (FRQ_BITS(CONV11_B_INT, w_type))
#define L11_BQ(val)   QMN(w_type, CONV11_B_FRAQ, val)

#define CONV11_OUT_FRAQ (FRQ_BITS(CONV11_OUT_INT, d_type))



// CONV12
//================================================
#define CONV12_W_SHAPE {16,32,1,2}
#define CONV12_W_ELEMENTS (16*32*1*2)
#define CONV12_W_RANK (4)

#define CONV12_W_FRAQ   (FRQ_BITS(CONV12_W_INT, w_type))
#define L12_WQ(val)   QMN(w_type, CONV12_W_FRAQ, val)

#define CONV12_B_ELEMENTS (16)
#define CONV12_B_SHAPE {16}
#define CONV12_B_RANK (1)

#define CONV12_B_FRAQ   (FRQ_BITS(CONV12_B_INT, w_type))
#define L12_BQ(val)   QMN(w_type, CONV12_B_FRAQ, val)

#define CONV12_OUT_FRAQ (FRQ_BITS(CONV12_OUT_INT, d_type))

// CONV13
//================================================
#define CONV13_W_SHAPE {16,16,1,2}
#define CONV13_W_ELEMENTS (16*16*1*2)
#define CONV13_W_RANK (4)

#define CONV13_W_FRAQ   (FRQ_BITS(CONV13_W_INT, w_type))
#define L13_WQ(val)   QMN(w_type, CONV13_W_FRAQ, val)

#define CONV13_B_ELEMENTS (16)
#define CONV13_B_SHAPE {16}
#define CONV13_B_RANK (1)

#define CONV13_B_FRAQ   (FRQ_BITS(CONV13_B_INT, w_type))
#define L13_BQ(val)   QMN(w_type, CONV13_B_FRAQ, val)

#define CONV13_OUT_FRAQ (FRQ_BITS(CONV13_OUT_INT, d_type))



// CONV14
//================================================
#define CONV14_W_SHAPE {8,16,1,2}
#define CONV14_W_ELEMENTS (8*16*1*2)
#define CONV14_W_RANK (4)

#define CONV14_W_FRAQ   (FRQ_BITS(CONV14_W_INT, w_type))
#define L14_WQ(val)   QMN(w_type, CONV14_W_FRAQ, val)

#define CONV14_B_ELEMENTS (8)
#define CONV14_B_SHAPE {8}
#define CONV14_B_RANK (1)

#define CONV14_B_FRAQ   (FRQ_BITS(CONV14_B_INT, w_type))
#define L14_BQ(val)   QMN(w_type, CONV14_B_FRAQ, val)

#define CONV14_OUT_FRAQ (FRQ_BITS(CONV14_OUT_INT, d_type))



// CONV15
//================================================
#define CONV15_W_SHAPE {8,16,1,2}
#define CONV15_W_ELEMENTS (8*16*1*2)
#define CONV15_W_RANK (4)

#define CONV15_W_FRAQ   (FRQ_BITS(CONV15_W_INT, w_type))
#define L15_WQ(val)   QMN(w_type, CONV15_W_FRAQ, val)

#define CONV15_B_ELEMENTS (8)
#define CONV15_B_SHAPE {8}
#define CONV15_B_RANK (1)

#define CONV15_B_FRAQ   (FRQ_BITS(CONV15_B_INT, w_type))
#define L15_BQ(val)   QMN(w_type, CONV15_B_FRAQ, val)

#define CONV15_OUT_FRAQ (FRQ_BITS(CONV15_OUT_INT, d_type))

// CONV16
//================================================
#define CONV16_W_SHAPE {8,8,1,2}
#define CONV16_W_ELEMENTS (8*8*1*2)
#define CONV16_W_RANK (4)

#define CONV16_W_FRAQ   (FRQ_BITS(CONV16_W_INT, w_type))
#define L16_WQ(val)   QMN(w_type, CONV16_W_FRAQ, val)

#define CONV16_B_ELEMENTS (8)
#define CONV16_B_SHAPE {8}
#define CONV16_B_RANK (1)

#define CONV16_B_FRAQ   (FRQ_BITS(CONV16_B_INT, w_type))
#define L16_BQ(val)   QMN(w_type, CONV16_B_FRAQ, val)

#define CONV16_OUT_FRAQ (FRQ_BITS(CONV16_OUT_INT, d_type))

// CONV17
//================================================
#define CONV17_W_SHAPE {8,8,1,2}
#define CONV17_W_ELEMENTS (8*8*1*2)
#define CONV17_W_RANK (4)

#define CONV17_W_FRAQ   (FRQ_BITS(CONV17_W_INT, w_type))
#define L17_WQ(val)   QMN(w_type, CONV17_W_FRAQ, val)

#define CONV17_B_ELEMENTS (8)
#define CONV17_B_SHAPE {8}
#define CONV17_B_RANK (1)

#define CONV17_B_FRAQ   (FRQ_BITS(CONV17_B_INT, w_type))
#define L17_BQ(val)   QMN(w_type, CONV17_B_FRAQ, val)

#define CONV17_OUT_FRAQ (FRQ_BITS(CONV17_OUT_INT, d_type))

// CONV18
//================================================
#define CONV18_W_SHAPE {4,8,1,2}
#define CONV18_W_ELEMENTS (4*8*1*2)
#define CONV18_W_RANK (4)

#define CONV18_W_FRAQ   (FRQ_BITS(CONV18_W_INT, w_type))
#define L18_WQ(val)   QMN(w_type, CONV18_W_FRAQ, val)

#define CONV18_B_ELEMENTS (4)
#define CONV18_B_SHAPE {4}
#define CONV18_B_RANK (1)

#define CONV18_B_FRAQ   (FRQ_BITS(CONV18_B_INT, w_type))
#define L18_BQ(val)   QMN(w_type, CONV18_B_FRAQ, val)

#define CONV18_OUT_FRAQ (FRQ_BITS(CONV18_OUT_INT, d_type))



// CONV19
//================================================
#define CONV19_W_SHAPE {4,8,1,2}
#define CONV19_W_ELEMENTS (4*8*1*2)
#define CONV19_W_RANK (4)

#define CONV19_W_FRAQ   (FRQ_BITS(CONV19_W_INT, w_type))
#define L19_WQ(val)   QMN(w_type, CONV19_W_FRAQ, val)

#define CONV19_B_ELEMENTS (4)
#define CONV19_B_SHAPE {4}
#define CONV19_B_RANK (1)

#define CONV19_B_FRAQ   (FRQ_BITS(CONV19_B_INT, w_type))
#define L19_BQ(val)   QMN(w_type, CONV19_B_FRAQ, val)

#define CONV19_OUT_FRAQ (FRQ_BITS(CONV19_OUT_INT, d_type))

// CONV20
//================================================
#define CONV20_W_SHAPE {4,4,1,2}
#define CONV20_W_ELEMENTS (4*4*1*2)
#define CONV20_W_RANK (4)

#define CONV20_W_FRAQ   (FRQ_BITS(CONV20_W_INT, w_type))
#define L20_WQ(val)   QMN(w_type, CONV20_W_FRAQ, val)

#define CONV20_B_ELEMENTS (4)
#define CONV20_B_SHAPE {4}
#define CONV20_B_RANK (1)

#define CONV20_B_FRAQ   (FRQ_BITS(CONV20_B_INT, w_type))
#define L20_BQ(val)   QMN(w_type, CONV20_B_FRAQ, val)

#define CONV20_OUT_FRAQ (FRQ_BITS(CONV20_OUT_INT, d_type))



// CONV21
//================================================
#define CONV21_W_SHAPE {2,4,1,2}
#define CONV21_W_ELEMENTS (2*4*1*2)
#define CONV21_W_RANK (4)

#define CONV21_W_FRAQ   (FRQ_BITS(CONV21_W_INT, w_type))
#define L21_WQ(val)   QMN(w_type, CONV21_W_FRAQ, val)

#define CONV21_B_ELEMENTS (2)
#define CONV21_B_SHAPE {2}
#define CONV21_B_RANK (1)

#define CONV21_B_FRAQ   (FRQ_BITS(CONV21_B_INT, w_type))
#define L21_BQ(val)   QMN(w_type, CONV21_B_FRAQ, val)

#define CONV21_OUT_FRAQ (FRQ_BITS(CONV21_OUT_INT, d_type))



// CONV22
//================================================
#define CONV22_W_SHAPE {2,4,1,2}
#define CONV22_W_ELEMENTS (2*4*1*2)
#define CONV22_W_RANK (4)

#define CONV22_W_FRAQ   (FRQ_BITS(CONV22_W_INT, w_type))
#define L22_WQ(val)   QMN(w_type, CONV22_W_FRAQ, val)

#define CONV22_B_ELEMENTS (2)
#define CONV22_B_SHAPE {2}
#define CONV22_B_RANK (1)

#define CONV22_B_FRAQ   (FRQ_BITS(CONV22_B_INT, w_type))
#define L22_BQ(val)   QMN(w_type, CONV22_B_FRAQ, val)

#define CONV22_OUT_FRAQ (FRQ_BITS(CONV22_OUT_INT, d_type))

// CONV23
//================================================
#define CONV23_W_SHAPE {2,2,1,2}
#define CONV23_W_ELEMENTS (2*2*1*2)
#define CONV23_W_RANK (4)

#define CONV23_W_FRAQ   (FRQ_BITS(CONV23_W_INT, w_type))
#define L23_WQ(val)   QMN(w_type, CONV23_W_FRAQ, val)

#define CONV23_B_ELEMENTS (2)
#define CONV23_B_SHAPE {2}
#define CONV23_B_RANK (1)

#define CONV23_B_FRAQ   (FRQ_BITS(CONV23_B_INT, w_type))
#define L23_BQ(val)   QMN(w_type, CONV23_B_FRAQ, val)

#define CONV23_OUT_FRAQ (FRQ_BITS(CONV23_OUT_INT, d_type))

// CONV24
//================================================
#define CONV24_W_SHAPE {2,2,1,2}
#define CONV24_W_ELEMENTS (2*2*1*2)
#define CONV24_W_RANK (4)

#define CONV24_W_FRAQ   (FRQ_BITS(CONV24_W_INT, w_type))
#define L24_WQ(val)   QMN(w_type, CONV24_W_FRAQ, val)

#define CONV24_B_ELEMENTS (2)
#define CONV24_B_SHAPE {2}
#define CONV24_B_RANK (1)

#define CONV24_B_FRAQ   (FRQ_BITS(CONV24_B_INT, w_type))
#define L24_BQ(val)   QMN(w_type, CONV24_B_FRAQ, val)

#define CONV24_OUT_FRAQ (FRQ_BITS(CONV24_OUT_INT, d_type))

// CONV25
//================================================
#define CONV25_W_SHAPE {2,2,1,1}
#define CONV25_W_ELEMENTS (2*2*1*1)
#define CONV25_W_RANK (4)

#define CONV25_W_FRAQ   (FRQ_BITS(CONV25_W_INT, w_type))
#define L25_WQ(val)   QMN(w_type, CONV25_W_FRAQ, val)

#define CONV25_B_ELEMENTS (2)
#define CONV25_B_SHAPE {2}
#define CONV25_B_RANK (1)

#define CONV25_B_FRAQ   (FRQ_BITS(CONV25_B_INT, w_type))
#define L25_BQ(val)   QMN(w_type, CONV25_B_FRAQ, val)

#define CONV25_OUT_FRAQ (FRQ_BITS(CONV25_OUT_INT, d_type))

#endif  //_EMNIST_CONSTANTS_H_