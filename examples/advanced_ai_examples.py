"""
Advanced AI Examples - Prompt templates, Chain of Thought, Self-reflection.

Demonstrates prompt engineering, multi-step reasoning, and response critique.
"""

import asyncio
from opensable.core.advanced_ai import PromptLibrary, ChainOfThought, SelfReflection


async def main():
    """Run advanced AI examples."""

    print("=" * 60)
    print("Advanced AI Examples")
    print("=" * 60)

    # Example 1: Prompt template library
    print("\n1. Prompt Template Library")
    print("-" * 40)

    library = PromptLibrary()

    templates = library.list_templates()
    print(f"Available templates: {len(templates)}")
    for name in templates[:5]:
        template = library.get(name)
        print(f"  - {name}: {template.description if template else ''}")

    # Example 2: Using templates
    print("\n2. Using Prompt Templates")
    print("-" * 40)

    # Summarize template
    summary_prompt = library.render(
        "summarize",
        text="Open-Sable is a production-ready AI assistant framework. It includes multi-agent orchestration, 11 chat interfaces, advanced skills like RAG and code execution, enterprise features with RBAC, and comprehensive monitoring.",
        max_words="25",
    )

    print("Summarize template:")
    print(f"{summary_prompt}\n")

    # Translate template
    translate_prompt = library.render(
        "translate", text="Hello, how are you?", source_lang="English", target_lang="Spanish"
    )

    print("Translate template:")
    print(f"{translate_prompt}\n")

    # Example 3: Code review template
    print("\n3. Code Review Template")
    print("-" * 40)

    code_review = library.render(
        "code_review",
        code="""
def calculate(x, y):
    return x + y
    
result = calculate(5, 3)
print(result)
""",
        language="python",
        focus_areas="correctness, performance, style",
    )

    print("Code review prompt generated")
    print(f"Length: {len(code_review)} chars\n")

    # Example 4: Custom template
    print("\n4. Custom Template Creation")
    print("-" * 40)

    from core.advanced_ai import PromptTemplate

    custom = PromptTemplate(
        name="bug_report",
        template="Report a bug in {component}:\n\nSteps: {steps}\nExpected: {expected}\nActual: {actual}",
        variables=["component", "steps", "expected", "actual"],
        description="Create a bug report",
    )

    library.add(custom)
    print(f"Added custom template: {custom.name}")

    bug_prompt = library.render(
        "bug_report",
        component="File Manager",
        steps="1. Upload file\n2. Try to download",
        expected="File downloads successfully",
        actual="404 error",
    )

    print(f"Bug report:\n{bug_prompt}\n")

    # Example 5: Chain of Thought reasoning
    print("\n5. Chain of Thought Reasoning")
    print("-" * 40)

    cot = ChainOfThought()

    question = "If I have 3 apples and buy 2 more, then give away half, how many do I have?"

    result = await cot.reason(question, max_steps=4)

    if result.success:
        print(f"Question: {question}")
        print(f"\nReasoning steps: {len(result.steps)}")
        for step in result.steps:
            print(f"\n  Step {step.step_number}: {step.question}")
            print(f"  Reasoning: {step.reasoning}")
            print(f"  Answer: {step.answer}")
            print(f"  Confidence: {step.confidence:.2f}")

        print(f"\nFinal Answer: {result.final_answer}")
        print(f"Overall Confidence: {result.confidence:.2f}")

    # Example 6: Complex reasoning
    print("\n6. Complex Multi-Step Problem")
    print("-" * 40)

    complex_question = "A company has 100 employees. 60% work remotely. Of remote workers, 75% prefer async communication. How many employees prefer async?"

    result = await cot.reason(complex_question, max_steps=5)

    if result.success:
        print(f"Question: {complex_question}")
        print(f"\nSteps taken: {len(result.steps)}")
        print(f"Final answer: {result.final_answer}")
        print(f"Confidence: {result.confidence:.2f}")

    # Example 7: Self-reflection
    print("\n7. Self-Reflection and Critique")
    print("-" * 40)

    reflection = SelfReflection()

    original_question = "What is Python?"
    original_response = "Python is a programming language. It's used for many things."

    result = await reflection.reflect(
        question=original_question,
        response=original_response,
        criteria=["Completeness", "Technical accuracy", "Clarity", "Examples"],
    )

    print(f"Original response: {original_response}")
    print("\nCritique:")
    print(f"{result.critique[:200]}...")
    print(f"\nImprovements suggested: {len(result.improvements)}")
    for imp in result.improvements[:3]:
        print(f"  - {imp}")
    print(f"\nQuality score: {result.quality_score:.2f}/1.0")

    # Example 8: Iterative improvement
    print("\n8. Iterative Response Improvement")
    print("-" * 40)

    question = "Explain machine learning"
    response_v1 = "Machine learning is when computers learn things."

    print(f"Version 1: {response_v1}")

    # First reflection
    result1 = await reflection.reflect(question, response_v1)
    print(f"Quality v1: {result1.quality_score:.2f}")

    # Improved response
    response_v2 = result1.revised_response

    # Second reflection
    result2 = await reflection.reflect(question, response_v2)
    print(f"Quality v2: {result2.quality_score:.2f}")
    print(f"Improvement: {((result2.quality_score - result1.quality_score) * 100):.1f}%")

    # Example 9: Template categories
    print("\n9. Template Organization")
    print("-" * 40)

    # Add metadata to templates
    from core.advanced_ai import PromptTemplate

    data_template = PromptTemplate(
        name="analyze_data",
        template="Analyze this dataset: {data}\nFocus on: {aspects}",
        variables=["data", "aspects"],
        description="Data analysis prompt",
        metadata={"category": "analytics", "domain": "data_science"},
    )

    library.add(data_template)

    # List by category
    analytics_templates = library.list_templates(category="analytics")
    print(f"Analytics templates: {analytics_templates}")

    # Example 10: Export/import templates
    print("\n10. Template Export/Import")
    print("-" * 40)

    export_path = "/tmp/opensable_prompts.json"
    library.export_all(export_path)
    print(f"Exported {len(templates)} templates to {export_path}")

    # Create new library and import
    new_library = PromptLibrary(storage_dir="/tmp/opensable_prompts_new")
    new_library.import_from(export_path)
    print("Imported templates into new library")

    # Example 11: Chain of thought with context
    print("\n11. Chain of Thought with Context")
    print("-" * 40)

    context = "The company uses Python, TypeScript, and Go for development."
    question = "Which language should we use for a new microservice?"

    result = await cot.reason(question, max_steps=3, context=context)

    if result.success:
        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"Final recommendation: {result.final_answer}")

    # Example 12: Few-shot learning template
    print("\n12. Few-Shot Learning Template")
    print("-" * 40)

    few_shot = PromptTemplate(
        name="sentiment_few_shot",
        template="""Classify sentiment as positive, negative, or neutral.

Examples:
- "I love this product!" -> positive
- "This is terrible." -> negative  
- "It's okay." -> neutral

Now classify: {text}""",
        variables=["text"],
        description="Sentiment classification with examples",
        examples=[
            {"text": "Great experience!", "output": "positive"},
            {"text": "Not good at all", "output": "negative"},
        ],
    )

    library.add(few_shot)

    sentiment_prompt = library.render("sentiment_few_shot", text="This is absolutely amazing!")

    print("Few-shot prompt created for sentiment analysis")

    print("\n" + "=" * 60)
    print("✅ Advanced AI examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
