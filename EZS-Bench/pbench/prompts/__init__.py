"""Prompt templates for VQA evaluation."""

from .evaluation_prompt import (
    system_template_binary_v0,
    begin_user_template_binary_v0,
    user_template_binary_v0,
    video_template_fn_v0,
    output_format_fn_binary_v0,
    templates
)

from .qa_text2world import (
    system_template_v0,
    user_template_fn_v0,
    user_template_fn_v1,
    output_format_v0,
    templates as qa_templates
)

__all__ = [
    'system_template_binary_v0',
    'begin_user_template_binary_v0',
    'user_template_binary_v0',
    'video_template_fn_v0',
    'output_format_fn_binary_v0',
    'templates',
    'system_template_v0',
    'user_template_fn_v0',
    'user_template_fn_v1',
    'output_format_v0',
    'qa_templates'
]
