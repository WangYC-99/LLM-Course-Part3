import re

src_file = open('/zhangpai25/wyc/glm_finetune/answers.txt', 'r')
answers = src_file.read().split('\n')
src_file.close()

results = []
for each_answer in answers:
    pattern = re.compile(r'.*B:(.*)')
    try:
        results.append(pattern.findall(each_answer)[0])
    except:
        continue
    # print(pattern.findall(each_answer))

tgt_file = open('output.txt', 'a')
for each_line in results:
    tgt_file.write(each_line + '\n')
tgt_file.close()
        