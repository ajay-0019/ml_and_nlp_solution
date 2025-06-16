import pandas as pd
import numpy as np

def get_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'
    
def pandas_filter_pass(dataframe):

    return dataframe[dataframe['Grade'].isin(['A', 'B'])]


names = ['Aarav', 'Saanvi', 'Vivaan', 'Diya', 'Aditya', 'Ananya', 'Kabir', 'Ishita', 'Aryan', 'Meera']
subjects = ['Math', 'Physics', 'Chemistry', 'Math', 'Physics', 'Chemistry', 'Math', 'Physics', 'Chemistry', 'Math']

scores = np.random.randint(50, 101, size=10)

df = pd.DataFrame({
    'Name': names,
    'Subject': subjects,
    'Score': scores,
    'Grade': [''] * 10
})
df['Grade'] = df['Score'].apply(get_grade)

sorted_df = df.sort_values(by='Score', ascending=False)
print(sorted_df)

average_scores = df.groupby('Subject')['Score'].mean()
print(average_scores)

passed_df = pandas_filter_pass(df)
print(passed_df)
