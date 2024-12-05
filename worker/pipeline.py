class PipelineStep:
    def process(self, data):
        raise NotImplementedError("Subclasses should implement this!")

class LayoutAnalysisStep(PipelineStep):
    def process(self, data):
        # Dummy implementation
        return data

class PDFToTextStep(PipelineStep):
    def process(self, data):
        # Dummy implementation
        return "Dummy text extracted from PDF"

class TextCleaningStep(PipelineStep):
    def process(self, data):
        # Dummy implementation
        return data.strip()

class TextToMarkdownStep(PipelineStep):
    def process(self, data):
        # Dummy implementation
        return "# Dummy Markdown\n\n" + data


def run_pipeline(pdf_path):
    data = pdf_path  # Starting data is the path to the PDF

    steps = [
        LayoutAnalysisStep(),
        PDFToTextStep(),
        TextCleaningStep(),
        TextToMarkdownStep(),
    ]

    for step in steps:
        data = step.process(data)

    return data  # Final Markdown content

