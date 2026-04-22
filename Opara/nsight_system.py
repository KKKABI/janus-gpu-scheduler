import torch
from transformers import BertTokenizer, BertForMaskedLM
from flash_attn import flash_attn_func
import time

# 设置模型和分词器
model_path = '/public_0/YPY/PROJECTS/model/bert-large'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path)

# 将模型移到GPU，如果GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 创建虚拟数据：假设输入的ID是已经过分词器处理的
input_ids = torch.tensor([[101, 2023, 2003, 1037, 3076, 3793, 102]])  # 对应于一些虚拟文本的token IDs（可以随意设置）

# 将输入数据移动到与模型相同的设备
inputs = {'input_ids': input_ids.to(device)}

# 使用FlashAttention进行前向计算
def flash_attention_forward(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
    """
    使用FlashAttention对Q, K, V张量执行注意力计算。
    这是一个简化版本，假设Q, K, V已经被正确形状化。
    """
    return flash_attn_func(q, k, v, dropout_p, softmax_scale, causal)

# 修改BERT的注意力层，使用flash_attention_forward代替默认的attention计算
class FlashBertForMaskedLM(BertForMaskedLM):
    def forward(self, **kwargs):
        # 获取BERT的输入
        input_ids = kwargs.get('input_ids')
        attention_mask = kwargs.get('attention_mask', None)

        # 先执行BERT的token化过程（embedding层）
        embedding_output = self.embeddings(input_ids)

        # 获取BERT的注意力层
        attention_output = self.encoder.layer[0].attention.self

        # 提取 Q, K, V
        query = attention_output.query(embedding_output)
        key = attention_output.key(embedding_output)
        value = attention_output.value(embedding_output)

        # 使用FlashAttention计算注意力
        attn_output = flash_attention_forward(query, key, value)

        # 继续后续的BERT处理（此处只是简化，实际操作会有更多层）
        outputs = self.encoder.layer[0].attention.output(attn_output)

        return outputs

# 用自定义的FlashBert模型进行推理
model = FlashBertForMaskedLM.from_pretrained(model_path)
model.to(device)

# 执行推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取BERT模型的输出logits
logits = outputs.logits
predicted_ids = torch.argmax(logits, dim=-1)

# 解码预测的ID为文本
predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

# 打印预测结果
print("虚拟输入的token IDs: ", input_ids)
print("预测文本: ", predicted_text)
