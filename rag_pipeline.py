from utils.pii_utils import redact_pii
from utils.llm_utils import generate_answer
from guardrails import Guard

# Guardrails setup
guard = Guard.from_rail("guardrails/guardrails_qa.xml")

def run_guarded_rag(user_query, context):
    # Step 1: Redact user input and context
    clean_query = redact_pii(user_query)
    clean_context = redact_pii(context)

    # Step 2: Generate answer
    raw_answer = generate_answer(clean_context, clean_query)

    # Step 3: Validate output with Guardrails
    response, validation = guard(
        prompt_params={
            "user_query": clean_query,
            "context": clean_context,
            "answer": raw_answer
        }
    )

    return response, validation