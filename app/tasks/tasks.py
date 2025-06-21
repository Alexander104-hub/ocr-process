import os
from detector import gt_generator
import luigi
import json
from pathlib import Path

from luigi_logger import logger
from handwritting import ImagePolygonProcessor, handwritting_model


class GenerateGroundTruth(luigi.Task):
    """Task to generate ground truth from image using CRAFT."""

    image_path = luigi.Parameter(default="test/1.jpg")
    data_dir = luigi.Parameter(default="test")
    text_threshold = luigi.FloatParameter(default=0.7)
    link_threshold = luigi.FloatParameter(default=0.4)
    low_text = luigi.FloatParameter(default=0.4)
    output_dir = luigi.Parameter(default="./test")

    def requires(self):
        return None

    def output(self):
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        filepath = Path(f"{self.output_dir}/{base_name}_gt.txt")
        return luigi.LocalTarget(filepath)

    def run(self):
        print("Generating ground truth file...")

        gt_path, polygons = gt_generator.generate_from_detection(
            self.image_path,
            output_dir=self.output_dir,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text,
        )


class ProcessImage(luigi.Task):
    """Task to process the image, extract polygons and recognize text."""

    image_path = luigi.Parameter(default="test/1.jpg")
    model_path = luigi.Parameter(default="model/my_model.h5")
    tmp_dir = luigi.Parameter(default="tmp")
    output_dir = luigi.Parameter(default="./test")
    id = luigi.Parameter(default="")

    def requires(self):
        return GenerateGroundTruth(
            image_path=self.image_path, output_dir=self.output_dir
        )

    def output(self):
        return luigi.LocalTarget(f"output/{self.id}_results.json")

    def run(self):
        logger.info("Processing image...")
        gt_path = self.input().path

        processor = ImagePolygonProcessor(tmp_dir=self.tmp_dir)

        results = processor.extract_polygons(
            self.image_path, gt_path, recognize_text=True, model=handwritting_model
        )

        serializable_results = []
        for i, result in enumerate(results):
            serializable_result = {
                "region_id": i + 1,
                "text": result.get("text", "No text recognized"),
                "polygon": result.get("polygon", []).tolist()
                if hasattr(result.get("polygon", []), "tolist")
                else result.get("polygon", []),
                "image_saved_path": None,
            }

            serializable_results.append(serializable_result)

        with self.output().open("w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Found {len(results)} text regions:")
        for i, result in enumerate(results):
            logger.info(f"Region {i+1}: {result.get('text', 'No text recognized')}")

        logger.info(f"Processed images saved to: {os.path.abspath('output')}")
        logger.info(f"Results saved to: {self.output().path}")


class OCRPipeline(luigi.WrapperTask):
    """The main pipeline task that runs everything."""

    image_path = luigi.Parameter(default="test/1.jpg")
    model_path = luigi.Parameter(default="model/my_model.h5")
    output_dir = luigi.Parameter(default="./test")
    id = luigi.Parameter(default="")

    def requires(self):
        return ProcessImage(
            image_path=self.image_path,
            model_path=self.model_path,
            output_dir=self.output_dir,
            id=self.id,
        )

    def output(self):
        return self.requires().output()
