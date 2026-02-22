"""
Advanced AI - Prompt Templates, Chain of Thought, and Self-Reflection.

Features:
- Prompt template management
- Variable substitution and formatting
- Chain of Thought reasoning
- Self-reflection and critique
- Prompt optimization
- Few-shot learning
"""

import asyncio
import json
import re
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


class PromptType(Enum):
    """Types of prompts."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class PromptTemplate:
    """Template for generating prompts."""

    name: str
    template: str
    variables: List[str] = field(default_factory=list)
    type: PromptType = PromptType.USER
    description: str = ""
    examples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs) -> str:
        """Render template with variables."""
        rendered = self.template

        for var in self.variables:
            if var in kwargs:
                placeholder = f"{{{var}}}"
                rendered = rendered.replace(placeholder, str(kwargs[var]))

        return rendered

    def validate(self, **kwargs) -> bool:
        """Check if all required variables are provided."""
        return all(var in kwargs for var in self.variables)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "variables": self.variables,
            "type": self.type.value,
            "description": self.description,
            "examples": self.examples,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        data = data.copy()
        if "type" in data:
            data["type"] = PromptType(data["type"])
        return cls(**data)


@dataclass
class ChainOfThoughtStep:
    """Single step in chain of thought reasoning."""

    step_number: int
    question: str
    reasoning: str
    answer: str
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "question": self.question,
            "reasoning": self.reasoning,
            "answer": self.answer,
            "confidence": self.confidence,
        }


@dataclass
class ChainOfThoughtResult:
    """Result from chain of thought reasoning."""

    success: bool
    final_answer: str = ""
    steps: List[ChainOfThoughtStep] = field(default_factory=list)
    reasoning_path: str = ""
    confidence: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "final_answer": self.final_answer,
            "steps": [step.to_dict() for step in self.steps],
            "reasoning_path": self.reasoning_path,
            "confidence": self.confidence,
            "error": self.error,
        }


@dataclass
class ReflectionResult:
    """Result from self-reflection."""

    original_response: str
    critique: str
    improvements: List[str]
    revised_response: str
    quality_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_response": self.original_response,
            "critique": self.critique,
            "improvements": self.improvements,
            "revised_response": self.revised_response,
            "quality_score": self.quality_score,
        }


class PromptLibrary:
    """
    Library of reusable prompt templates.

    Features:
    - Store and retrieve templates
    - Variable substitution
    - Template categories
    - Import/export templates
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize prompt library.

        Args:
            storage_dir: Directory for storing templates
        """
        self.storage_dir = (
            Path(storage_dir) if storage_dir else Path.home() / ".opensable" / "prompts"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.templates: Dict[str, PromptTemplate] = {}
        self._load_templates()
        self._load_default_templates()

    def _load_templates(self):
        """Load templates from storage."""
        for file in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(file.read_text())
                template = PromptTemplate.from_dict(data)
                self.templates[template.name] = template
            except Exception:
                pass

    def _load_default_templates(self):
        """Load default prompt templates."""
        defaults = [
            PromptTemplate(
                name="summarize",
                template="Summarize the following text in {max_words} words:\n\n{text}",
                variables=["text", "max_words"],
                description="Summarize text to specified length",
                examples=[{"text": "Long article...", "max_words": "50"}],
            ),
            PromptTemplate(
                name="translate",
                template="Translate the following text from {source_lang} to {target_lang}:\n\n{text}",
                variables=["text", "source_lang", "target_lang"],
                description="Translate text between languages",
            ),
            PromptTemplate(
                name="code_review",
                template="Review the following {language} code and provide feedback:\n\n```{language}\n{code}\n```\n\nFocus on: {focus_areas}",
                variables=["code", "language", "focus_areas"],
                description="Review code with specific focus areas",
            ),
            PromptTemplate(
                name="explain_like_im_five",
                template="Explain {concept} in simple terms that a 5-year-old would understand.",
                variables=["concept"],
                description="Explain complex concepts simply",
            ),
            PromptTemplate(
                name="brainstorm",
                template="Generate {num_ideas} creative ideas for {topic}. Consider: {constraints}",
                variables=["topic", "num_ideas", "constraints"],
                description="Brainstorm ideas with constraints",
            ),
            PromptTemplate(
                name="debug",
                template="Debug this {language} code:\n\n```{language}\n{code}\n```\n\nError: {error}\n\nProvide the fix and explanation.",
                variables=["code", "language", "error"],
                description="Debug code with error message",
            ),
            PromptTemplate(
                name="analyze_sentiment",
                template="Analyze the sentiment of this text and classify as positive, negative, or neutral:\n\n{text}",
                variables=["text"],
                description="Sentiment analysis",
            ),
        ]

        for template in defaults:
            if template.name not in self.templates:
                self.templates[template.name] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        return self.templates.get(name)

    def add(self, template: PromptTemplate):
        """Add template to library."""
        self.templates[template.name] = template
        self._save_template(template)

    def remove(self, name: str):
        """Remove template from library."""
        if name in self.templates:
            del self.templates[name]
            template_file = self.storage_dir / f"{name}.json"
            if template_file.exists():
                template_file.unlink()

    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """List all template names."""
        if category:
            return [
                name
                for name, template in self.templates.items()
                if template.metadata.get("category") == category
            ]
        return list(self.templates.keys())

    def render(self, name: str, **kwargs) -> Optional[str]:
        """Render template with variables."""
        template = self.get(name)
        if template and template.validate(**kwargs):
            return template.render(**kwargs)
        return None

    def _save_template(self, template: PromptTemplate):
        """Save template to storage."""
        template_file = self.storage_dir / f"{template.name}.json"
        template_file.write_text(json.dumps(template.to_dict(), indent=2))

    def export_all(self, output_file: str):
        """Export all templates to file."""
        data = {name: template.to_dict() for name, template in self.templates.items()}
        Path(output_file).write_text(json.dumps(data, indent=2))

    def import_from(self, input_file: str):
        """Import templates from file."""
        data = json.loads(Path(input_file).read_text())
        for name, template_data in data.items():
            template = PromptTemplate.from_dict(template_data)
            self.add(template)


class ChainOfThought:
    """
    Chain of Thought reasoning for complex problems.

    Breaks down complex questions into steps and reasons through each.
    """

    def __init__(self, llm_function: Optional[Callable] = None):
        """
        Initialize CoT reasoning.

        Args:
            llm_function: Function to call LLM (async)
        """
        self.llm_function = llm_function

    async def reason(
        self, question: str, max_steps: int = 5, context: Optional[str] = None
    ) -> ChainOfThoughtResult:
        """
        Perform chain of thought reasoning.

        Args:
            question: Question to answer
            max_steps: Maximum reasoning steps
            context: Additional context

        Returns:
            ChainOfThoughtResult with steps and final answer
        """
        steps = []
        current_question = question
        reasoning_path = []

        # Build initial prompt
        system_prompt = """You are an expert at step-by-step reasoning. 
Break down complex problems into clear steps.
For each step:
1. Identify the sub-question
2. Provide detailed reasoning
3. Give a clear answer
4. Assess your confidence (0-1)"""

        if context:
            system_prompt += f"\n\nContext: {context}"

        try:
            for step_num in range(1, max_steps + 1):
                # Generate step
                step_prompt = f"""Question: {current_question}

Think step-by-step. What is the next logical step to answer this question?

Provide your response in this format:
STEP {step_num}:
Question: [the sub-question for this step]
Reasoning: [your detailed reasoning]
Answer: [your answer to this step]
Confidence: [0.0 to 1.0]"""

                # Call LLM if available
                if self.llm_function:
                    response = await self.llm_function(step_prompt, system=system_prompt)
                else:
                    # Simulated response for testing
                    response = f"""STEP {step_num}:
Question: Break down the problem
Reasoning: This is step {step_num} of the reasoning process
Answer: Partial answer for step {step_num}
Confidence: 0.8"""

                # Parse response
                step = self._parse_step(step_num, response)
                steps.append(step)
                reasoning_path.append(f"Step {step_num}: {step.answer}")

                # Check if we have final answer
                if "final answer" in step.answer.lower() or step_num == max_steps:
                    break

                # Update question for next step
                current_question = f"Given: {step.answer}\nWhat's the next step?"

            # Generate final answer
            final_answer = steps[-1].answer if steps else "Unable to determine answer"
            avg_confidence = sum(s.confidence for s in steps) / len(steps) if steps else 0.0

            return ChainOfThoughtResult(
                success=True,
                final_answer=final_answer,
                steps=steps,
                reasoning_path="\n".join(reasoning_path),
                confidence=avg_confidence,
            )

        except Exception as e:
            return ChainOfThoughtResult(success=False, error=str(e))

    def _parse_step(self, step_num: int, response: str) -> ChainOfThoughtStep:
        """Parse LLM response into step."""
        # Extract components
        question_match = re.search(r"Question:\s*(.+?)(?=\n|Reasoning:)", response, re.DOTALL)
        reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=\n|Answer:)", response, re.DOTALL)
        answer_match = re.search(r"Answer:\s*(.+?)(?=\n|Confidence:|$)", response, re.DOTALL)
        confidence_match = re.search(r"Confidence:\s*([\d.]+)", response)

        question = question_match.group(1).strip() if question_match else f"Step {step_num}"
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5

        return ChainOfThoughtStep(
            step_number=step_num,
            question=question,
            reasoning=reasoning,
            answer=answer,
            confidence=confidence,
        )


class SelfReflection:
    """
    Self-reflection and critique for improving responses.

    Analyzes responses and suggests improvements.
    """

    def __init__(self, llm_function: Optional[Callable] = None):
        """
        Initialize self-reflection.

        Args:
            llm_function: Function to call LLM (async)
        """
        self.llm_function = llm_function

    async def reflect(
        self, question: str, response: str, criteria: Optional[List[str]] = None
    ) -> ReflectionResult:
        """
        Reflect on and improve a response.

        Args:
            question: Original question
            response: Response to critique
            criteria: Evaluation criteria

        Returns:
            ReflectionResult with critique and improvements
        """
        criteria = criteria or ["Accuracy", "Completeness", "Clarity", "Relevance", "Helpfulness"]

        critique_prompt = f"""Original Question: {question}

Original Response: {response}

Critique this response based on these criteria: {', '.join(criteria)}

Provide:
1. A detailed critique
2. Specific improvements
3. A revised response
4. A quality score (0-10)

Format your response as:
CRITIQUE:
[your critique]

IMPROVEMENTS:
- [improvement 1]
- [improvement 2]
...

REVISED RESPONSE:
[improved response]

QUALITY SCORE: [0-10]"""

        try:
            # Call LLM if available
            if self.llm_function:
                result = await self.llm_function(critique_prompt)
            else:
                # Simulated response
                result = """CRITIQUE:
The response is generally good but could be more detailed.

IMPROVEMENTS:
- Add more specific examples
- Improve structure and formatting
- Include relevant citations

REVISED RESPONSE:
[Improved version of the original response with better details and structure]

QUALITY SCORE: 8"""

            # Parse result
            critique = self._extract_section(result, "CRITIQUE")
            improvements = self._extract_improvements(result)
            revised = self._extract_section(result, "REVISED RESPONSE")
            score_match = re.search(r"QUALITY SCORE:\s*([\d.]+)", result)
            score = float(score_match.group(1)) / 10 if score_match else 0.8

            return ReflectionResult(
                original_response=response,
                critique=critique,
                improvements=improvements,
                revised_response=revised,
                quality_score=score,
            )

        except Exception as e:
            return ReflectionResult(
                original_response=response,
                critique=f"Error during reflection: {e}",
                improvements=[],
                revised_response=response,
                quality_score=0.0,
            )

    def _extract_section(self, text: str, section: str) -> str:
        """Extract a section from response."""
        pattern = rf"{section}:\s*(.+?)(?=\n[A-Z]+:|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_improvements(self, text: str) -> List[str]:
        """Extract improvement list."""
        section = self._extract_section(text, "IMPROVEMENTS")
        improvements = re.findall(r"[-•]\s*(.+)", section)
        return [imp.strip() for imp in improvements]


# Example usage
async def main():
    """Example advanced AI features."""

    print("=" * 50)
    print("Advanced AI Examples")
    print("=" * 50)

    # Prompt Templates
    print("\n1. Prompt Templates")
    library = PromptLibrary()

    # List templates
    templates = library.list_templates()
    print(f"  Available templates: {len(templates)}")
    print(f"  Templates: {', '.join(templates[:5])}")

    # Use template
    prompt = library.render(
        "summarize", text="This is a long article about AI and machine learning...", max_words="50"
    )
    print(f"\n  Rendered prompt:\n  {prompt[:100]}...")

    # Chain of Thought
    print("\n2. Chain of Thought Reasoning")
    cot = ChainOfThought()

    result = await cot.reason("What is 15% of 240?", max_steps=3)

    if result.success:
        print(f"  Question answered in {len(result.steps)} steps")
        print(f"  Final answer: {result.final_answer}")
        print(f"  Confidence: {result.confidence:.2f}")
        print("\n  Reasoning path:")
        for step in result.steps:
            print(f"    Step {step.step_number}: {step.question}")

    # Self-Reflection
    print("\n3. Self-Reflection")
    reflection = SelfReflection()

    original_response = "Python is a programming language. It's used for many things."

    reflection_result = await reflection.reflect(
        question="What is Python?", response=original_response
    )

    print(f"  Original: {original_response}")
    print(f"  Critique: {reflection_result.critique[:100]}...")
    print(f"  Improvements: {len(reflection_result.improvements)} suggested")
    print(f"  Quality score: {reflection_result.quality_score:.2f}")

    print("\n✅ Advanced AI examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
