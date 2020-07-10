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


剩下的部分因為太過零散還沒整個連接起來打通，所以在下一次書審再繳交
