### t5-onnx模型用于中文拼写纠错
t5纠错模型及配置可自行下载 -> https://huggingface.co/shibing624/mengzi-t5-base-chinese-correction， 放到项目中。

原始模型文件组成：
```
mengzi-t5-base-chinese-correction
|-- config.json
|-- pytorch_model.bin
|-- special_tokens_map.json
|-- spiece.model
|-- tokenizer_config.json
`-- tokenizer.json
```

#### t5纠错模型转为onnx格式，并进行量化，进一步减小模型大小，提高推理效率
```
def convert_to_onnx_model(model_path, model_onnx_path):
    '''
    :param model_path: 原始t5模型
    :param model_onnx_path: 保存为Onnx格式的路径
    :return:
    '''
    # 将t5模型转为onnx格式，并进行量化，以快速推断和减少模型大小
    model = export_and_get_onnx_model(model_path, model_onnx_path, quantized=True)
```

转换为onnx并量化后的模型文件：

```
T5是一个seq2seq模型(Encoder-Decoder)，由于它反复使用decoder进行推断，我们不能直接将整个模型导出到onnx。
我们需要分别导出编码器和解码器。

|-- mengzi-t5-base-chinese-correction-decoder-quantized.onnx
|-- mengzi-t5-base-chinese-correction-encoder-quantized.onnx
|-- mengzi-t5-base-chinese-correction-init-decoder-quantized.onnx
```

#### 加载onnx模型并进行纠错
```
def correct_test(model_path, model_onnx_path):
    model = get_onnx_model(model_path, model_onnx_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    t_input = '真麻烦你了。希望你们好好的跳无'
    token = tokenizer(t_input, return_tensors='pt')
    tokens = model.generate(input_ids=token['input_ids'],
                            attention_mask=token['attention_mask'],
                            num_beams=2)
    output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
    corrected_text, sub_details = get_errors(output, t_input)
    print("original sentence:{} => {} err:{}".format(t_input, corrected_text, sub_details))

    result:
            original sentence:真麻烦你了。希望你们好好的跳无 => 真麻烦你了。希望你们好好的跳舞 err:[('无', '舞', 14, 15)]
```


#### reference
https://github.com/shibing624/pycorrector
