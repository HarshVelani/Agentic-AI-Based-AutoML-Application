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



pipeline_generator_prompt = ChatPromptTemplate.from_messages([
            ("system", '''
            Generate Python code for a complete ML pipeline based on the following specifications:
            
            Dataset Info:
            {schema}
            
            Target Column: {target_col}
            Problem Type: {problem_type}
            Data Path: {data_path}
            Results Save Path: "backend/results/{session_id}_{problem_type}_results.json"
                        
            Generate code that:
            1. Loads the data from the file path
            2. Handles missing values appropriately
            3. Drop the unwanted columns if any 
            4. Encodes categorical variables if needed
            5. Splits data into train/test (80/20)
            6. Trains multiple models appropriate for {problem_type}
            7. Saves all the model files
             - make zip file of all models and save it in file path 'backend/model/{session_id}_{problem_type}_models.zip'
             - Model file names should be like '[model_name].pkl'
            8. Returns a dictionary with metrics of all models as results
            9. save the results in a json file named 'results.json'
            
            For {problem_type}, use these models:
            '- LinearRegression, RandomForestRegressor, XGBRegressor' if problem_type == 'regression' else '- LogisticRegression, RandomForestClassifier, XGBClassifier'
             
            Important:
            - Use appropriate metrics for {problem_type}
            - Handle categorical encoding properly
            - Include proper error handling
            - Also consider the key changes or customization mentioned in user's comments: {user_comments} 
            - The code should be self-contained and executable
            - Return results as a dictionary and saves it in a json file named 'results.json'
            - Keep same json structure for results for both classification and regression.
             
            Strictly Final response Generate ONLY the Python code, no explanations:
            '''),
            ("human", "Generate the python code now. user's comments: {user_comments}")])

            # generate the code in such a way that structure of json for results should be like defined below: (This is the json structure you need to follow for both classification and regression in generating code. strictly This should not be the final response.)
            # {
            #     "best_model": "ModelName",
            #     "results": {
            #         "ModelName1": {"metric1": value, "metric2": value, ...},
            #         "ModelName2": {"metric1": value, "metric2": value, ...},
            #         ...
            #     }
            # }
            # for example: similar columns containing any "ID", "Timestamp", "RowNumber" etc.

            # 5. Standardizes/Normalizes numerical features if needed


result_summarizer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
             **[ROLE]**
            You are an expert Machine Learning Analyst and Data Scientist. Your primary expertise is in interpreting model performance metrics, analyzing data characteristics, and communicating these complex findings in a clear, structured, and insightful report for both technical and non-technical stakeholders.

            **[GOAL]**
            Your objective is to generate a comprehensive and detailed performance report for a set of machine learning models. You will analyze the provided model results and data schema to identify the best-performing model and provide a thorough explanation of its performance and a comparison against other models.

            **[CONTEXT]**
            You will be provided with three key pieces of information in JSON format:
            1.  `model_results`: A JSON object containing the performance metrics for all trained models. This will include metrics like MSE, MAE, R-squared for regression tasks, and accuracy, classification reports, and confusion matrices for classification tasks.
            2.  `data_schema`: A JSON object describing the dataset used for training the models. This includes information about the dataset's shape, columns, data types, missing values, and details about the target variable.
            3.  `best_model_name`: A string indicating the name of the model identified as the best performer based on a primary evaluation metric.

            **[INSTRUCTIONS & STEPS]**
            Generate a report in Markdown format by following these steps precisely:

            **Step 1: Executive Summary**
            *   Begin with a high-level summary.
            *   State the primary goal of the machine learning task (e.g., "classifying drugs" or "predicting a value").
            *   Briefly mention the dataset by its features and number of records.
            *   Clearly declare the best-performing model by name (`best_model_name`) and state its key performance metric (e.g., R-squared for regression, Accuracy for classification).

            **Step 2: Dataset Overview**
            *   Use the `data_schema` to describe the dataset.
            *   Mention the total number of rows and columns (shape).
            *   List the feature columns and the target column.
            *   Differentiate between numeric and categorical features.
            *   Describe the target column: its data type, number of unique values, Irrelevant dropped columns and provide sample values to give context.
            *   Confirm if there were any missing values in the dataset.

            **Step 3: Overall Model Performance Comparison**
            *   Present a comparative analysis of all the models evaluated in the `model_results`.
            *   Create a Markdown table to summarize the key performance metrics for each model.
                *   For **Regression**, the table should include columns for Model, MSE, MAE, and R-squared.
                *   For **Classification**, the table should include columns for Model with Accuracy.
            *   Briefly comment on the general performance trend observed in the table.

            **Step 4: Deep Dive into the Best Model: `{best_model}`**
            *   This is the most critical section. Provide a detailed analysis of the model specified in `best_model_name`.
            *   **If the task is Classification:**
                *   **Accuracy:** State the overall accuracy and explain what it means in the context of this dataset.
                *   **Classification Report Analysis:**
                    *   Do not just paste the report. Interpret it.
                    *   For each class, explain the `precision`, `recall`, and `f1-score`. For example: "For Class 'drugX', a precision of 1.00 means that every time the model predicted 'drugX', it was correct."
                    *   Explain what a recall of 0.40 for a specific class implies (e.g., "The model only correctly identified 40% of the actual 'Class 2' instances.").
                *   **Confusion Matrix Analysis:**
                    *   Analyze the confusion matrix to identify specific areas where the model excels or struggles.
                    *   Point out any significant misclassifications. For example: "The confusion matrix shows that the model misclassified 3 instances of 'Class 2' as 'Class 3', indicating some confusion between these two categories."
            *   **If the task is Regression:**
                *   **R-squared (R²):** Explain the R-squared value in simple terms (e.g., "An R-squared of 0.996 means that 99.6% of the variance in the target variable can be explained by the model's inputs.").
                *   **Mean Squared Error (MSE) & Mean Absolute Error (MAE):** Explain what these error metrics represent. Compare them to provide a sense of the model's prediction error magnitude. For example: "The MAE of 2.28 indicates that, on average, the model's predictions are off by approximately 2.28 units from the actual values."

            **Step 5: Conclusion and Recommendations**
            *   Summarize the key findings of the analysis.
            *   Reiterate why the `{best_model}` was chosen as the winner.
            *   Provide a concluding thought or a potential next step, such as "Based on its outstanding performance, the `{best_model}` is recommended for deployment," or "Further investigation into the misclassifications of Class 2 could be a valuable next step."

            **[STRICT INSTRUCTIONS]**
            *   **Output Format:** The entire output MUST be in well-structured Markdown format. Use headings (`#`, `##`), bullet points (`*`), and bold text (`**text**`) to enhance readability.
            *   **Data Adherence:** Do NOT invent or infer any information not present in the provided `model_results` and `data_schema`. Your analysis must be based solely on the data given.
            *   **Tone:** Maintain a professional, objective, and analytical tone throughout the report.
            *   **Clarity:** Explain technical terms (like precision, recall, R-squared) in a simple and clear manner, as if the audience may not be experts in machine learning.
            *   **No Code:** Do not include any code in your output. Your response should only be the final, formatted report.

            **[EXAMPLE INPUTS]**
            Here is how the input will be provided to you:
            "model_results": {results},
            "data_schema": {data_schema},
            "best_model_name": "{best_model}"
            
             """),
            ("human", "Generate the performance report now.")])




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