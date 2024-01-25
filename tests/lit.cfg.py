import os

from lit.formats import ShTest
from lit.llvm import llvm_config

config.name = "foo"
config.test_format = ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.foo_tools_dir]
tools = ['foo-opt']
llvm_config.add_tool_substitutions(tools, tool_dirs)
