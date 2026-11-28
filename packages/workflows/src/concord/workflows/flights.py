from datetime import date

from hatchet_sdk import Hatchet, Context
from pydantic import BaseModel

# TODO: What configuration is needed to make this work?
hatchet = Hatchet()


AFCA_BATCH_SLOT_COUNT = 35
MERGE_TARGET_SLOT_COUNT = 288


class FlightAdvectionDayInput(BaseModel):
    """Input model for the flight advection day workflow."""

    date: str
    # Add other fields as needed (scene cadence?)


flight_advection_day_workflow = hatchet.workflow(
    name="flight_advection_day_workflow",
    input_validator=FlightAdvectionDayInput
)


@flight_advection_day_workflow.task()
def prepare_inputs(input_data: FlightAdvectionDayInput, ctx: Context):
    """Prepare inputs for the flight advection day workflow."""
    # Implement the logic to prepare inputs
    return {"prepared_data": input_data}
