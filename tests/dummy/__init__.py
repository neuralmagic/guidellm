"""
The tests.dummy package package represents dummy data factories and test services.

test.dummy.data.OpenAIModel - openai.Model test factory
test.dummy.data.OpenAICompletion - openai.Completion test factory
test.dummy.data.OpenAICompletionChoice - openai.CompletionChoice test factory

test.dummy.services.TestRequestGenerator - RequestGenerator that is used
    for testing purposes
"""

from . import data, services
