# Distill_ner_model
对roberta-base-finetuned-cluener2020-chinese预训练模型做模型蒸馏并用onnx加速模型推理  
目前尝试的方法有：  
  1、student_model模型层：一层embedding+两层transformer+一层linear+一层crf。  
    loss函数：crf_loss+mse_loss(预训练模型bert输出层的值,学生模型的linear层输出)  
    效果：acc：0.92  
    评价：效果很好，但是crf层无法用onnx加速。（目前还未找到解决办法，只能去掉crf层试试）  
  2、student_model模型层：一层embedding+两层transformer+一层linear  
    loss函数：直接用linear输出学teacher模型的linear层标签分类概率。
    效果：acc：0.48  
    评价：嗯效果不行，还是要对transformer的输出做限制。  
  3、student_model模型层：一层embedding+两层transformer+一层linear  
     loss函数：cross_entropy(学生模型标签预测值，真实值)+mse_loss(预训练模型bert输出层的值,学生模型的transformer层输出)  
     效果：acc：0.8823  
     评价：效果比方法1差一些。想想怎么整。  
     如果只用cross_entropy(学生模型标签预测值，真实值)的话：
     效果：acc：0.7864（30个epoch未收敛时）=>那证明mse_loss还是有用的。  
     方法3转onnx后推理速度：推理“北京朝阳区北苑华贸城”这句话时间是：0.0061168670654296875 秒
  4、student_model模型层：一层embedding+两层transformer+一层linear 
     loss函数：cross_entropy(学生模型标签预测值，教师标签值)+mse_loss(预训练模型bert输出层的值,学生模型的transformer层输出)  
     
     
     
    
     
