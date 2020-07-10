# 2020_ARC_
目前已完成的部分:
1. deep model for gait analysis
2. 手機APP程式(因為只會用藍芽跟ARC IoTDK連接所以沒有上傳到這裡)
3. ARC IoTDK的藍芽測試
4. ARC IoTDK的sensor資料蒐集
5. ARC IoTDK的MLI範例測試

目前還正在進行中：
1. 用tensorflow lite 測試，因為之前的MLI問題很多(src裡有附上用MLI轉換出來的code)，所以目前改用tflite測試
2. 將sensor蒐集到的資料跟tflie結合
3. 將iotdk的藍芽跟已經寫好的手機app連上。

已上傳的部分:
1. model for gait analysis ( torch_test0.0625_3layer_6axis.ipynb )
2. model for stride length analysis ( model_runwalk_SL_multitask_full_20Hz.py )
3. trained model for stride length analysis ( Multitask_early_swing_20Hzrandom5_super_parttotest_fullpath_com0.125_allpeople_+weiMAX.pkl )
4. deep model with C language by MLI ( src folder )


剩下的部分因為太過零散還沒整個連接起來打通，所以在比賽當天完整呈現

# Introduction
本作品將步態分析深度模型應用在iotdk，使用iotdk的感測器所蒐集到的資料作為模型輸入，在將分析結果以藍芽傳至手機以視覺化，實現了物聯網與即時監控的功能。

# HW/SW Setup
1. iotdk 開發環境
2. iotdk 開發版
3. tensorflow > 2.2
4. android smartphone ( 即時監控APP，不支援ios )

# User manual
這部分會在作品整個完成後詳細說明，但目前預計使用方式應該只需要用metaware development tool kit 啟動板子就好。
