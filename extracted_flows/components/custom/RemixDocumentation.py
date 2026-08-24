# Recovered Langflow component
# type: RemixDocumentation
# class: RemixDocumentation
# used in 1 flow(s): NVIDIA RTX Remix
# json path: node.data.node.template.code.value

import httpx
import json
import re
from langflow.custom import Component
from langflow.io import Output, MessageTextInput
from langflow.schema import DataFrame, Data


class RemixDocumentation(Component):
    display_name = "RTX Remix Documentation"
    description = "Fetch information from the NVIDIA RTX Remix documentation"
    documentation: str = (
        "https://docs.omniverse.nvidia.com/kit/docs/rtx_remix/latest/index.html"
    )
    icon = "NVIDIA"
    name = "RemixDocumentation"

    inputs = [
        MessageTextInput(
            name="exclude_paths",
            display_name="Exclude Paths",
            info=(
                "List of path prefixes to exclude from results. "
                "Used to filter out sections of the documentation that are not typically relevant to user queries."
            ),
            is_list=True,
            value=[
                "source",
                "CHANGELOG.html",
                "docs/changelog",
                "docs/contributing/api.html",
            ],
        ),
    ]

    outputs = [
        Output(
            display_name="DataFrame",
            name="dataframe_output",
            method="fetch_documentation_dataframe",
        ),
        Output(
            display_name="Data", name="data_output", method="fetch_documentation_data"
        ),
    ]

    _BASE_URL = "https://docs.omniverse.nvidia.com/kit/docs/rtx_remix/latest/"

    def _fetch_all_documentation(self) -> list[Data]:
        """Fetch all documentation entries and return as list of Data objects."""
        # URL for the search index JavaScript file
        search_index_url = self._BASE_URL + "searchindex.js"

        # Fetch the search index file
        response = httpx.get(search_index_url, follow_redirects=True)
        response.raise_for_status()

        # Extract the JSON data from the JavaScript file
        # The file contains: const searchData = {...};
        js_content = response.text

        # Extract the JSON part using regex
        match = re.search(r"const searchData = ({.*});", js_content, re.DOTALL)
        if not match:
            raise ValueError("Could not parse search index data")

        # Parse the JSON data
        search_data = json.loads(match.group(1))

        # Get the data array which contains all searchable items
        data_items = search_data.get("data", [])

        results = []
        for item in data_items:
            filename = item.get("filename", "")
            text = item.get("content", "")

            if not text:
                continue

            # Check if filename should be excluded
            should_exclude = False
            for exclude_path in self.exclude_paths:
                if filename.startswith(exclude_path):
                    should_exclude = True
                    break

            if should_exclude:
                continue

            # Build URL with proper handling of empty anchors
            anchor = item.get("anchor", "")
            if anchor:
                url = f"{self._BASE_URL}{filename}#{anchor}"
            else:
                url = f"{self._BASE_URL}{filename}"

            results.append(
                Data(
                    title=item.get("display_name", ""),
                    text=text,
                    url=url,
                )
            )

        # If no results found, create a single entry indicating no results
        if not results:
            data = Data(
                title="No documentation found",
                text="No documentation entries were found in the search index",
                url=search_index_url,
            )
            results.append(data)

        return results

    def fetch_documentation_dataframe(self) -> DataFrame:
        """Fetch documentation and return as DataFrame."""
        results = self._fetch_all_documentation()
        data_frame = DataFrame(results)
        self.status = data_frame
        return data_frame

    def fetch_documentation_data(self) -> list[Data]:
        """Fetch documentation and return as list of Data objects."""
        results = self._fetch_all_documentation()
        self.status = results
        return results
