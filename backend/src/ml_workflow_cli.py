from src.workflow_manager import AgenticMLWorkflow
import os
from typing import Dict, Any
from dotenv import load_dotenv  
load_dotenv()  # Load environment variables from .env file


# Interactive CLI Interface
class MLWorkflowCLI:
    """Command-line interface for the ML workflow"""
    
    def __init__(self):
        self.workflow = AgenticMLWorkflow()
    
    def run_interactive(self):
        """Run interactive CLI"""
        print("🤖 Agentic ML Workflow")
        print("=" * 50)
        
        while True:
            try:
                # Get user inputs
                data_path = "data/Churn_Modelling.csv"
                data_path = "data/drug200.csv"# input("\n📁 Enter path to your data file (CSV/Excel): ").strip()

                if not data_path:
                    break
                
                target_column = "EstimatedSalary" # input("🎯 Enter target column name: ").strip()
                target_column = "Drug"


                problem_type = "" # input("🔍 Enter problem type (Classification/Regression) or press Enter to auto-detect: ").strip().lower()
                
                if not problem_type:
                    problem_type = None
                
                tune_choice = "n" # input("⚡ Do you want hyperparameter tuning? (y/n): ").strip().lower()
                tune_model = tune_choice in ['y', 'yes']

                user_comments = "Predict based on the given data and its features." # input("📝 (Optional) Describe your dataset/problem: ").strip()
                
                print("\n🚀 Running ML workflow...")
                
                # Run workflow
                results = self.workflow.run_workflow(
                    data_path=data_path,
                    target_column=target_column,
                    problem_type=problem_type,
                    tune_model=tune_model,
                    user_comments=user_comments
                )
                
                # Display results
                self._display_results(results)
                
                # Ask if user wants to continue
                continue_choice = input("\n🔄 Run another workflow? (y/n): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    break
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            # except Exception as e:
            #     print(f"\n❌ Error: {e}")
    
    def _display_results(self, results: Dict[str, Any]):
        """Display workflow results"""
        print("\n" + "="*50)
        print("📊 WORKFLOW RESULTS")
        print("="*50)
        
        print(f"🎯 Problem Type: {results['problem_type']}")
        
        print("\n📈 Performance Metrics:")
        # for metric, value in results['metrics'].items():
        #     if isinstance(value, float):
        #         print(f"   {metric}: {value:.4f}")
        #     else:
        #         print(f"   {metric}: {value}")
        
        
        for model_name, value in results['metrics'].items():
            print(f"\n\n{model_name}:")
            for metric, val in value.items():
                print(f"\n         {metric}: {val}")


        print(f"\n💾 Model saved at: {results['model_path']}")
        
        print("\n✅ Process Log:")
        for message in results['messages']:
            print(f"   {message}")
        
        if results['errors']:
            # print("\n❌ Errors:")
            for error in results['errors']:
                print(f"   {error}")
        
        print("\n📋 Dataset Info:")
        schema = results['data_schema']
        if schema:
            print(f"   Shape: {schema.get('shape', 'N/A')}")
            print(f"   Columns: {len(schema.get('columns', []))}")
            print(f"   Missing values: {sum(schema.get('missing_values', {}).values())}")




if __name__ == "__main__":
    # Run interactive CLI
    cli = MLWorkflowCLI()
    cli.run_interactive()