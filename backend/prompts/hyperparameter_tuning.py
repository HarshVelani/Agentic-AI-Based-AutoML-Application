from langchain_core.prompts import ChatPromptTemplate

hyperparameter_tuning_prompt = ChatPromptTemplate.from_messages([
            ("system", '''
            Generate Python code for hyperparameter tuning for a {problem_type} problem.
            
            Requirements:
            1. Load the same data and preprocessing from before
            2. Use GridSearchCV or RandomizedSearchCV for tuning
            3. Focus on the best performing model from the previous run
            4. Define appropriate parameter grids
            5. Use 5-fold cross-validation
            6. Save the tuned model as 'tuned_model.pkl'
            7. Return updated metrics
            
            Generate ONLY the Python code, no explanations:
            '''),
            ("human", "Generate the hyperparameter tuning code now.")])