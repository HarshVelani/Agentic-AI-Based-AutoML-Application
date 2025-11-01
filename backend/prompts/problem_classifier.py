from langchain_core.prompts import ChatPromptTemplate

problem_classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", '''
                prompt = f"""
                Analyze the target column and determine if this is a Classification or Regression problem.
                
                Target column info:
                {target_info}
                
                Rules:
                - If target is numeric with many unique values (>20) → Regression
                - If target is categorical or numeric with few unique values (≤20) → Classification
                - Consider the nature of the data
                
                Respond with only one word: "classification" or "regression"
                '''),
            ("human", "Classify the problem type now.")])