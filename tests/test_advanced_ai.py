"""
Tests for Advanced AI features - Prompts, Chain of Thought, Self-reflection.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from opensable.core.advanced_ai import (
    PromptLibrary, PromptTemplate, PromptType,
    ChainOfThought, SelfReflection
)


class TestPromptLibrary:
    """Test prompt template library"""
    
    @pytest.fixture
    def library(self):
        return PromptLibrary()
    
    def test_get_template(self, library):
        """Test getting a template"""
        template = library.get("summarize")
        
        assert template is not None
        assert template.name == "summarize"
        assert len(template.variables) > 0
    
    def test_render_template(self, library):
        """Test rendering a template with variables"""
        result = library.render(
            "summarize",
            text="Long text here",
            max_words="50"
        )
        
        assert "Long text here" in result
        assert "50" in result
    
    def test_list_templates(self, library):
        """Test listing all templates"""
        templates = library.list_templates()
        
        assert len(templates) > 0
        assert "summarize" in templates
        assert "translate" in templates
    
    def test_add_custom_template(self, library):
        """Test adding custom template"""
        custom = PromptTemplate(
            name="custom_test",
            template="Test {var1} and {var2}",
            variables=["var1", "var2"],
            description="Test template"
        )
        
        library.add(custom)
        
        retrieved = library.get("custom_test")
        assert retrieved.name == "custom_test"
    
    def test_template_with_metadata(self, library):
        """Test template metadata"""
        template = PromptTemplate(
            name="meta_test",
            template="Test",
            variables=[],
            metadata={"category": "test", "version": "1.0"}
        )
        
        library.add(template)
        
        retrieved = library.get("meta_test")
        assert retrieved.metadata["category"] == "test"
    
    def test_export_import(self, library, tmp_path):
        """Test template export and import"""
        export_path = tmp_path / "templates.json"
        
        # Export
        library.export_all(str(export_path))
        assert export_path.exists()
        
        # Import to new library
        new_library = PromptLibrary(storage_dir=str(tmp_path / "new"))
        new_library.import_from(str(export_path))
        
        # Verify templates imported
        assert len(new_library.list_templates()) > 0


class TestChainOfThought:
    """Test Chain of Thought reasoning"""
    
    @pytest.fixture
    def cot(self):
        return ChainOfThought()
    
    @pytest.mark.asyncio
    async def test_basic_reasoning(self, cot):
        """Test basic reasoning"""
        result = await cot.reason(
            "If I have 5 apples and buy 3 more, how many do I have?",
            max_steps=3
        )
        
        assert result.success
        assert len(result.steps) > 0
        assert result.final_answer is not None
        assert result.confidence > 0
    
    @pytest.mark.asyncio
    async def test_multi_step_reasoning(self, cot):
        """Test multi-step problem"""
        result = await cot.reason(
            "A store has 100 items. 30% are sold in morning, 40% in afternoon. How many left?",
            max_steps=5
        )
        
        assert result.success
        assert len(result.steps) >= 2  # Should have multiple steps
    
    @pytest.mark.asyncio
    async def test_reasoning_with_context(self, cot):
        """Test reasoning with additional context"""
        result = await cot.reason(
            "Which framework should we use?",
            max_steps=3,
            context="We need: Python, async support, REST API"
        )
        
        assert result.success
        assert result.final_answer is not None
    
    @pytest.mark.asyncio
    async def test_confidence_scores(self, cot):
        """Test confidence scoring"""
        result = await cot.reason("Simple math: 2 + 2", max_steps=2)
        
        assert result.success
        # Each step should have confidence
        for step in result.steps:
            assert 0 <= step.confidence <= 1


class TestSelfReflection:
    """Test self-reflection and critique"""
    
    @pytest.fixture
    def reflection(self):
        return SelfReflection()
    
    @pytest.mark.asyncio
    async def test_basic_reflection(self, reflection):
        """Test basic response critique"""
        result = await reflection.reflect(
            question="What is Python?",
            response="Python is a programming language.",
            criteria=["Completeness", "Accuracy"]
        )
        
        assert result.critique is not None
        assert len(result.improvements) > 0
        assert 0 <= result.quality_score <= 1
    
    @pytest.mark.asyncio
    async def test_improvement_suggestions(self, reflection):
        """Test improvement suggestions"""
        result = await reflection.reflect(
            question="Explain machine learning",
            response="ML is when computers learn.",
            criteria=["Detail", "Examples", "Technical accuracy"]
        )
        
        # Should suggest improvements for vague response
        assert len(result.improvements) > 0
        assert result.quality_score <= 0.8  # Low score for vague answer
    
    @pytest.mark.asyncio
    async def test_revised_response(self, reflection):
        """Test that revised response is generated"""
        result = await reflection.reflect(
            question="What is an API?",
            response="It's a thing.",
            criteria=["Clarity", "Detail"]
        )
        
        assert result.revised_response is not None
        # Revised should be longer/better
        assert len(result.revised_response) > len("It's a thing.")
    
    @pytest.mark.asyncio
    async def test_quality_scoring(self, reflection):
        """Test quality score calculation"""
        # Poor response
        poor = await reflection.reflect(
            "Explain AI",
            "AI is stuff",
            criteria=["Completeness", "Accuracy", "Detail"]
        )
        
        # Better response
        good = await reflection.reflect(
            "Explain AI",
            "Artificial Intelligence is the simulation of human intelligence by machines, involving learning, reasoning, and problem-solving.",
            criteria=["Completeness", "Accuracy", "Detail"]
        )
        
        # Both should produce valid quality scores
        assert 0 <= poor.quality_score <= 1.0
        assert 0 <= good.quality_score <= 1.0
