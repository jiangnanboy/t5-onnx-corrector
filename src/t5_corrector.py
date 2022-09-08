from fastT5 import export_and_get_onnx_model, get_onnx_model
from transformers import AutoTokenizer
import operator
import os

unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'

def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if i >= len(corrected_text):
            continue
        if ori_char in unk_tokens:
            # deal with unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if ori_char != corrected_text[i]:
            if not is_chinese(ori_char):
                # pass not chinese char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            if not is_chinese(corrected_text[i]):
                corrected_text = corrected_text[:i] + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details

def convert_to_onnx_model(model_path, model_onnx_path):
    '''
    :param model_path: 原始我们t5模型
    :param model_onnx_path: 保存为Onnx格式的路径
    :return:
    '''
    # 将t5模型转为onnx格式，并进行量化，以快速推断和减少模型大小
    model = export_and_get_onnx_model(model_path, model_onnx_path, quantized=True)

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

if __name__ == '__main__':
    base_path = os.path.abspath('..')
    model_path = os.path.join(base_path, 'mengzi-t5-base-chinese-correction')

    base_path = os.path.abspath('.')
    model_onnx_path = os.path.join(base_path, 'models')
    correct_test(model_path, model_onnx_path)