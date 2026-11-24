from pathlib import Path
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# load the gemini key from the .env file in the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# which gemini model to use change it here if google updates the name
GEMINI_MODEL = "gemini/gemini-3.5-flash"


# takes the patients answers and their result and gives back the ai explanation as text
def explain_result(input_data, result):

    # turn the form numbers back into words so the ai understands them
    levels = {1: "normal", 2: "above normal", 3: "well above normal"}
    yes_no = {0: "no", 1: "yes"}

    if input_data["is_male"] == 1:
        sex = "male"
    else:
        sex = "female"

    patient = (
        f"Age: {input_data['age_years']} years\n"
        f"Sex: {sex}\n"
        f"Height: {input_data['height']} cm\n"
        f"Weight: {input_data['weight']} kg\n"
        f"Blood pressure: {input_data['ap_hi']}/{input_data['ap_lo']} mmHg\n"
        f"Cholesterol: {levels[input_data['cholesterol']]}\n"
        f"Glucose: {levels[input_data['gluc']]}\n"
        f"Smokes: {yes_no[input_data['smoke']]}\n"
        f"Drinks alcohol: {yes_no[input_data['alco']]}\n"
        f"Physically active: {yes_no[input_data['active']]}"
    )

    # try means if anything goes wrong like no internet or a bad key
    # the website keeps working and just shows no ai explanation instead of crashing
    try:
        llm = LLM(
            model=GEMINI_MODEL,
            api_key=os.getenv("GEMINI_API_KEY"),
        )

        # the agent is the ai worker its role goal and backstory tell it how to behave
        explainer = Agent(
            role="Heart Health Results Explainer",
            goal="Explain a heart disease risk result to a patient in simple friendly English",
            backstory=(
                "You help people understand results from a heart disease risk tool. "
                "You explain things clearly and calmly without medical jargon. "
                "You never diagnose anyone and always remind people to see a doctor."
            ),
            llm=llm,
            allow_delegation=False,
        )

        # the task is the job we give the agent
        task = Task(
            description=(
                "A machine learning model estimated this person's risk of heart disease.\n\n"
                f"Their details:\n{patient}\n\n"
                f"Estimated risk: {result['probability']}% ({result['risk_level']} risk)\n\n"
                "Write a short explanation for this person. "
                "Say which of their details are commonly linked to higher or lower heart disease risk. "
                "Give two or three general healthy habits that could help. "
                "Do not diagnose them. Remind them this is not a medical diagnosis "
                "and they should talk to a doctor. "
                "Keep it under 150 words. Use plain text only with no bullet points or symbols."
            ),
            expected_output="A short plain English explanation under 150 words",
            agent=explainer,
        )

        # the crew runs the task and kickoff starts it
        crew = Crew(agents=[explainer], tasks=[task])
        output = crew.kickoff()
        return str(output)

    except Exception as error:
        # print the problem in the terminal so we can fix it
        print("AI explanation failed:", error)
        return None