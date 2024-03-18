# coding: utf-8

import openai
import pandas as pd
import time

# your ChatGPT API key
openai.api_key = ''  # Please replace with your API key.

def get_classification(prompt, temperature=0.7):
    messages = [{"role": "user", "content": content} for content in prompt]
    response = openai.ChatCompletion.create(
        #Replace here with the GPT version you are using.
        model="gpt-4",
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message['content']

data = pd.read_csv("suicide_all_LLM.csv", encoding="utf-8", sep='\t')



prompt_intro = ["自杀风险分级任务分为以下11级：\n 0：未见任何生存痛苦表达，只有日常负面情绪表达。1：生存痛苦有所表达，未见自杀愿望表达。2：生存痛苦已明确表达，未见自杀愿望表达。3：强烈的生存痛苦，未见自杀愿望表达。4：自杀愿望已表达，具体方式和计划未明。5：强烈的自杀愿望，自杀方式未明。6：自杀已经计划中，自杀日期未明。7：自杀方式已确定，自杀日期未明。8：自杀已经计划中，自杀日期大体确定。9：自杀方式已确定，近日内可能进行。10：自杀可能正在进行中。\n下方内容分别显示了用户的ID及其在社交媒体上的发文内容。请你进行自杀风险分级任务，判断以下发文内容属于上述11种级别中的哪一种级别。请你用markdown表格的形式输出分类结果，输出结果的格式如下：表头依次为：用户ID、自杀风险分级标签、推理原因。\n 注意：请一步一步进行思考。表格中的推理原因请尽量详细，要有原文的内容以及根据此内容推断的结果。只返回表格，其他内容不要有。"]

BATCH_SIZE = 1

#Replace here with the path to your result file.
with open('result-temperature0.7.txt', 'a') as result_file:
    table_header = "| id | 标签 |"
    table_divider = "|----|------|"
    result_file.write(table_header)
    result_file.write("\n")
    result_file.write(table_divider)
    result_file.write("\n")

    for start_index in range(0, len(data), BATCH_SIZE):
        batch = data.iloc[start_index:start_index + BATCH_SIZE]
        prompts_for_batch = []

        for index, row in batch.iterrows():
            prompt = [f"id: {row['id']} {row['comment']}"]
            prompts_for_batch.extend(prompt)

        complete_prompt = prompt_intro + prompts_for_batch
        print(complete_prompt)
        classification = get_classification(complete_prompt)

        results = classification.split("\n")[2:]
        print(results)
        for i, row in enumerate(batch.iterrows()):
            result_file.write(results[i] + "\n")

        time.sleep(20)

print("Finished!")
