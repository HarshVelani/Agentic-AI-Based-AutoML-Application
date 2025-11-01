from langchain_core.prompts import ChatPromptTemplate

pipeline_generator_prompt = ChatPromptTemplate.from_messages([
            ("system", '''
            Generate Python code for a complete ML pipeline based on the following specifications:
            
            Dataset Info:
            {schema}
            
            Target Column: {target_col}
            Problem Type: {problem_type}
            Data Path: {data_path}
            Results Save Path: "results/{session_id}_{problem_type}_results.json"
                        
            Generate code that:
            1. Loads the data from the file path
            2. Handles missing values appropriately
            3. Drop the unwanted columns if any 
            4. Encodes categorical variables if needed
            5. Splits data into train/test (80/20)
            6. Trains multiple models appropriate for {problem_type}
            7. Saves all the model files
             - make zip file of all models and save it in file path 'model/{session_id}_{problem_type}_models.zip'
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