"""Handler for /generate-architectural-review — invoked by the Step Function or webhook."""
import json
import logging

from activities.generate_impact_review import GenerateImpactReview
#Test Comment - Prod

class ProcessArchitecturalReviewRequest:
    def __init__(self, event):
        logging.info("Class Name: %s, Method Name: __init__", self.__class__.__name__)
        self.event = event

    def handle(self):
        try:
            logging.info("Class Name: %s, Method Name: handle", self.__class__.__name__)
            event = self._format_data(self.event)
            review = GenerateImpactReview(event)
            return review.run()
        except Exception as e:
            logging.error("ProcessArchitecturalReviewRequest error: %s", e, exc_info=True)
            raise

    def _format_data(self, data):
        if isinstance(data, dict) and "body" in data:
            body = data.get("body")
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except (TypeError, ValueError):
                    pass
            if isinstance(body, dict) and ("repo_object" in body or "pr_number" in body):
                return body
        return data
