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

# 定义一个提示的列表，您可以根据需求自定义

prompt_intro = ["自杀风险分级任务分为以下11级：\n 0：未见任何生存痛苦表达，只有日常负面情绪表达。1：生存痛苦有所表达，未见自杀愿望表达。2：生存痛苦已明确表达，未见自杀愿望表达。3：强烈的生存痛苦，未见自杀愿望表达。4：自杀愿望已表达，具体方式和计划未明。5：强烈的自杀愿望，自杀方式未明。6：自杀已经计划中，自杀日期未明。7：自杀方式已确定，自杀日期未明。8：自杀已经计划中，自杀日期大体确定。9：自杀方式已确定，近日内可能进行。10：自杀可能正在进行中。\n以上自杀风险分级的各个级别含义解释如下：0级（未见任何生存痛苦表达，只有日常负面情绪表达。）：检测到正面或中性情绪表达以及日常的负面情绪吐槽。缺乏生存痛苦的描述。1级（生存痛苦有所表达，未见自杀愿望表达）：表达了不适、悲伤或压力，但没有表达死亡或结束生命的愿望。2级（生存痛苦已明确表达，未见自杀愿望表达）：明确的生存痛苦表达，如绝望。提及生活质量受损，但没有提及死亡或自杀的想法。3级（强烈的生存痛苦，未见自杀愿望表达）：表达了强烈的绝望感，可能包含对未来没有希望的感受。使用了强烈负面词汇来描述当前的生活状况。4级（自杀愿望已表达，具体方式和计划未明）：表达了想死或不想活的想法，但没有说明具体的自杀计划或方法。5级（强烈的自杀愿望，自杀方式未明）：表达了迫切的死亡愿望，可能会说感觉无法继续下去。提到了自杀的想法，但没有提及具体如何实施。6级（自杀已经计划中，自杀日期未明）：提及了有关自杀的具体计划，例如留遗书、处理后事等。有对自杀行为的具体思考，但没有提到确切的时间。7级（自杀方式已确定，自杀日期未明）：描述了具体的自杀方法，如药物、高处跳下等。明确表述了自杀的意图，但未提及日期。8级（自杀已经计划中，自杀日期大体确定）：提及自杀行动将在不久的将来发生，可能会有时间范围的暗示。可能会提到与自杀相关的预备行动，如告别信或特别安排。9级（自杀方式已确定，近日内可能进行）：明确提到了在近期内的自杀意图和计划。可能会有计划的执行细节，如选定的日期和时间。10级（自杀可能正在进行中）：表达了即刻自杀的行动，如正在告别或已经处于危险情况中。可能包含紧急的求助信息或感觉是最后的告别。\n下方内容分别显示了用户的ID及其在社交媒体上的发文内容。请你进行自杀风险分级任务，判断以下发文内容属于上述11种级别中的哪一种级别。请你用markdown表格的形式输出分类结果，输出结果的格式如下：表头依次为：用户ID、自杀风险分级标签、推理原因。\n 注意：请一步一步进行思考。表格中的推理原因请尽量详细，请根据上述自杀风险分级的各个级别的含义解释进行分类判断。要有原文的内容以及根据此内容推断的结果。只返回表格，其他内容不要有。"]

BATCH_SIZE = 3

#Replace here with the path to your result file.
with open('result-temperature0.7-jibiejieshi.txt', 'a') as result_file:
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
