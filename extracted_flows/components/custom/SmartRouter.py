# Recovered Langflow component
# type: SmartRouter
# class: SmartRouterComponent
# used in 1 flow(s): AITGPT V1.0.0
# json path: node.data.node.template.code.value

from typing import Any

from lfx.custom import Component
from lfx.io import BoolInput, HandleInput, MessageInput, MessageTextInput, MultilineInput, Output, TableInput
from lfx.schema.message import Message
from lfx.schema.table import EditMode


class SmartRouterComponent(Component):
    display_name = "Smart Router"
    description = "Routes an input message using LLM-based categorization."
    icon = "route"
    name = "SmartRouter"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._matched_category = None
        self._classification_done = False

    inputs = [
        HandleInput(
            name="llm",
            display_name="Language Model",
            info="LLM to use for categorization.",
            input_types=["LanguageModel"],
            required=True,
        ),
        MessageTextInput(
            name="input_text",
            display_name="Input",
            info="The primary text input for the operation.",
            required=True,
        ),
        TableInput(
            name="routes",
            display_name="Routes",
            info=(
                "Define the categories for routing. Each row should have a route/category name "
                "and optionally a custom output value."
            ),
            table_schema=[
                {
                    "name": "route_category",
                    "display_name": "Route Name",
                    "type": "str",
                    "description": "Name for the route (used for both output name and category matching)",
                    "edit_mode": EditMode.INLINE,
                },
                {
                    "name": "route_description",
                    "display_name": "Route Description",
                    "type": "str",
                    "description": "Description of when this route should be used",
                    "default": "",
                    "edit_mode": EditMode.POPOVER,
                },
                {
                    "name": "output_value",
                    "display_name": "Route Message (Optional)",
                    "type": "str",
                    "description": (
                        "Optional message to send when this route is matched. "
                        "Leave empty to pass through the original input text."
                    ),
                    "default": "",
                    "edit_mode": EditMode.POPOVER,
                },
            ],
            real_time_refresh=True,
            required=True,
        ),
        MessageInput(
            name="message",
            display_name="Override Output",
            info="Optional override message for all routes.",
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="enable_else_output",
            display_name="Include Else Output",
            info="Include an Else output for cases that don't match any route.",
            value=False,
            advanced=True,
        ),
        MultilineInput(
            name="custom_prompt",
            display_name="Additional Instructions",
            info="Additional instructions for LLM-based categorization.",
            advanced=True,
        ),
    ]

    outputs: list[Output] = []

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        if field_name in {"routes", "enable_else_output"}:
            frontend_node["outputs"] = []
            routes_data = field_value if field_name == "routes" else getattr(self, "routes", [])

            for i, row in enumerate(routes_data):
                route_category = row.get("route_category", f"Category {i + 1}")
                frontend_node["outputs"].append(
                    Output(
                        display_name=route_category,
                        name=f"category_{i + 1}_result",
                        method=f"category_{i + 1}_response",
                        group_outputs=True,
                    )
                )

            enable_else = field_value if field_name == "enable_else_output" else getattr(self, "enable_else_output", False)
            if enable_else:
                frontend_node["outputs"].append(
                    Output(display_name="Else", name="default_result", method="default_response", group_outputs=True)
                )
        return frontend_node

    def _build_prompt(self, input_text: str, categories: list[dict]) -> str:
        category_info = []
        for i, category in enumerate(categories):
            cat_name = category.get("route_category", f"Category {i + 1}")
            cat_desc = category.get("route_description", "")
            if cat_desc and cat_desc.strip():
                category_info.append(f'"{cat_name}": {cat_desc}')
            else:
                category_info.append(f'"{cat_name}"')

        categories_text = "\n".join(f"- {info}" for info in category_info)

        base_prompt = (
            f'You are a text classifier.\n\n'
            f'Text to classify: "{input_text}"\n\n'
            f"Available categories:\n{categories_text}\n\n"
            f'Respond with ONLY the exact category name that best matches the text.\n'
            f'If none match well, respond with "NONE".'
        )

        custom_prompt = getattr(self, "custom_prompt", "")
        if custom_prompt and custom_prompt.strip():
            simple_routes = ", ".join(
                [f'"{cat.get("route_category", f"Category {i + 1}")}"' for i, cat in enumerate(categories)]
            )
            formatted_custom = custom_prompt.format(input_text=input_text, routes=simple_routes)
            return f"{base_prompt}\n\nAdditional Instructions:\n{formatted_custom}"

        return base_prompt

    def _classify_once(self):
        if self._classification_done:
            return

        self._classification_done = True
        self._matched_category = None

        categories = getattr(self, "routes", [])
        input_text = getattr(self, "input_text", "")
        llm = getattr(self, "llm", None)

        if not llm or not categories:
            self.status = "No LLM or routes provided"
            return

        prompt = self._build_prompt(input_text, categories)
        self.status = f"Prompt sent to LLM:\\n{prompt}"

        try:
            if hasattr(llm, "invoke"):
                response = llm.invoke(prompt)
                categorization = response.content.strip().strip('"') if hasattr(response, "content") else str(response).strip().strip('"')
            else:
                categorization = str(llm(prompt)).strip().strip('"')

            self.status = f"LLM response: '{categorization}'"

            for i, category in enumerate(categories):
                route_category = category.get("route_category", "")
                if categorization.lower() == route_category.lower():
                    self._matched_category = i
                    self.status = f"Matched category: {route_category}"
                    return

            self.status = f"No match found for '{categorization}'"

        except RuntimeError as e:
            self.status = f"Error in LLM categorization: {e!s}"

    def _build_output_message(self, idx: int) -> Message:
        categories = getattr(self, "routes", [])
        input_text = getattr(self, "input_text", "")

        override_output = getattr(self, "message", None)
        if override_output and hasattr(override_output, "text") and str(override_output.text).strip():
            return Message(text=str(override_output.text))
        if isinstance(override_output, str) and override_output.strip():
            return Message(text=override_output)

        custom_output = categories[idx].get("output_value", "")
        if custom_output and str(custom_output).strip() and str(custom_output).strip().lower() != "none":
            return Message(text=str(custom_output))

        return Message(text=input_text)

    def _route_response(self, idx: int) -> Message:
        self._classify_once()

        if self._matched_category == idx:
            return self._build_output_message(idx)

        self.stop(f"category_{idx + 1}_result")
        return Message(text="")

    def category_1_response(self) -> Message:
        return self._route_response(0)

    def category_2_response(self) -> Message:
        return self._route_response(1)

    def category_3_response(self) -> Message:
        return self._route_response(2)

    def category_4_response(self) -> Message:
        return self._route_response(3)

    def default_response(self) -> Message:
        enable_else = getattr(self, "enable_else_output", False)
        if not enable_else:
            self.stop("default_result")
            return Message(text="")

        self._classify_once()

        if self._matched_category is not None:
            self.stop("default_result")
            return Message(text="")

        override_output = getattr(self, "message", None)
        if override_output and hasattr(override_output, "text") and str(override_output.text).strip():
            return Message(text=str(override_output.text))
        if isinstance(override_output, str) and override_output.strip():
            return Message(text=override_output)

        return Message(text=getattr(self, "input_text", ""))