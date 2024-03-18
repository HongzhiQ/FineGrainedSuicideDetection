# coding: utf-8

import openai
import pandas as pd
import time

# Set API key
openai.api_key = ''  # Replace with your API key

# Read data
data = pd.read_csv('../../data/fine-grained/fold5.tsv', sep='\t')

# Functions that call the GPT-4 API
def call_gpt4_api(prompt, temperature=1.0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Replace with the GPT model version you are using
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message['content']


prompt_intro = ["请你进行数据生成任务，模仿中国社交媒体中在“微博”发言的用户生成有关自杀风险的留言。\n自杀风险分级任务分为以下11级：\n 0：未见任何生存痛苦表达，只有日常负面情绪表达。1：生存痛苦有所表达，未见自杀愿望表达。2：生存痛苦已明确表达，未见自杀愿望表达。3：强烈的生存痛苦，未见自杀愿望表达。4：自杀愿望已表达，具体方式和计划未明。5：强烈的自杀愿望，自杀方式未明。6：自杀已经计划中，自杀日期未明。7：自杀方式已确定，自杀日期未明。8：自杀已经计划中，自杀日期大体确定。9：自杀方式已确定，近日内可能进行。10：自杀可能正在进行中。\n请你根据自杀风险分级的各个级别，模仿中国社交网络中使用“微博”软件的用户的发言特点，进行数据生成工作。生成30条社交媒体上用户的发言。注意：用户发言的场景如下：在一名上吊自杀成功的用户（用户id为走饭）生前最后一条帖子（此帖子为准备自杀的帖子）底下进行评论，在此评论的用户大多都有一些自杀的风险，有的借此抒发自己的郁闷情绪，有人借此表达自己的自杀想法或自杀计划。只不过有的人是低自杀风险，有的人是高自杀风险。这些用户借此评论抒发自己的自杀意图、自杀想法、生存痛苦、自杀计划等。\n注意：生成用户的发言内容符合自杀风险级别中的第{label}级别！生成的用户发言数据请符合“微博”社交媒体发言风格。\n以下例子被专家标注为第{label}级别，请你学习下面的例子，找出内在关系，学习下面的例子为什么被标注为第{label}级别，模仿现有标注数据，仔细观察数据特点，生成相应数据并返回生成的30条文本数据!\n"]  # 您的 prompt


# Process each label level
for label in range(0,11):
    # Randomly select 6 pieces of data
    sample_data = data[data['myLabel'] == label].sample(6)
    generated_data = []
    # Call API for each piece of data
    allData = ''
    for comment in sample_data['comment']:
        allData = allData + comment +'\n'
    full_prompt = prompt_intro[0].format(label=label) + allData
    print(full_prompt)

    generated_comment = call_gpt4_api(full_prompt)
    print(generated_comment)
    generated_data.append([generated_comment, label])

    # Save generated data
    new_data_df = pd.DataFrame(generated_data, columns=['comment', 'myLabel'])
    new_data_df.to_csv(f'', sep='\t', index=False)
    time.sleep(20)

print("success！")