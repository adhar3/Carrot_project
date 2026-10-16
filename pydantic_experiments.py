"""
Pydantic v2 notes + experiments for the Carrot Field AI backend
================================================================

Run:   pip install "pydantic>=2"   then   python pydantic_experiments.py

QUICK NOTES
-----------
* A BaseModel is a class whose fields are type-checked when you create it.
  Bad input raises ValidationError (FastAPI turns that into an HTTP 422).
* Default ("lax") mode converts when it safely can: "12" -> 12.
  Strict mode refuses to convert.
* Field(...) adds rules: ge/le/gt/lt, min_length/max_length, pattern, etc.
* @field_validator checks or cleans ONE field. @model_validator checks
  rules that involve SEVERAL fields (e.g. latitude AND longitude).
* model_dump() = model -> dict. model_validate() = dict -> model.
  model_dump(exclude_unset=True) = only fields the caller actually sent
  (that's what a PUT/PATCH update endpoint needs).
* Pydantic v1 -> v2 renames: .dict() -> model_dump(), @validator ->
  @field_validator, class Config -> model_config, regex= -> pattern=,
  Field(example=...) -> Field(examples=[...]).
"""
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

Severity = Literal["none", "low", "moderate", "high"]


# ------------------------------------------------------------------ helpers
def show_errors(e: ValidationError) -> None:
    for err in e.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "(model)"
        print(f"       - {loc}: {err['msg']}  [type={err['type']}]")


def attempt(label: str, fn) -> None:
    """Run fn(); print the result or the validation errors."""
    try:
        print(f"  OK   {label}: {fn()!r}")
    except ValidationError as e:
        print(f"  FAIL {label}:")
        show_errors(e)


def header(title: str) -> None:
    print(f"\n=== {title} ===")


# ------------------------------------------------------------ experiments
class Obs(BaseModel):
    row_id: str
    carrots_inspected: int


def exp1_coercion() -> None:
    header("1. Type checking and coercion (lax mode)")
    attempt('"12" for an int', lambda: Obs(row_id="ROW-01", carrots_inspected="12"))
    # Expect: OK, carrots_inspected=12 (converted to int)
    attempt("12.0 for an int", lambda: Obs(row_id="ROW-01", carrots_inspected=12.0))
    # Expect: OK, 12
    attempt("12.7 for an int", lambda: Obs(row_id="ROW-01", carrots_inspected=12.7))
    # Expect: FAIL, a fractional number is not a valid integer
    attempt('"abc" for an int', lambda: Obs(row_id="ROW-01", carrots_inspected="abc"))
    # Expect: FAIL, not a valid integer
    attempt("missing field", lambda: Obs(row_id="ROW-01"))
    # Expect: FAIL, type=missing for carrots_inspected


class StrictObs(BaseModel):
    model_config = ConfigDict(strict=True)
    carrots_inspected: int


def exp2_strict_mode() -> None:
    header("2. Strict mode (no conversions)")
    attempt('"12" in strict mode', lambda: StrictObs(carrots_inspected="12"))
    # Expect: FAIL, strict mode will not turn a string into an int
    attempt("12 in strict mode", lambda: StrictObs(carrots_inspected=12))
    # Expect: OK
    # Question to think about: should API payloads be strict? (JSON already has
    # real numbers, but HTML forms and query strings only send strings.)


class Row(BaseModel):
    row_id: str = Field(pattern=r"^ROW-\d{2}$")
    severity: Severity
    carrots_inspected: int = Field(ge=0)
    notes: str = Field(default="", max_length=200)


def exp3_field_rules() -> None:
    header("3. Field rules: pattern, Literal, ge, max_length")
    attempt("valid", lambda: Row(row_id="ROW-02", severity="low", carrots_inspected=5))
    attempt("bad row_id", lambda: Row(row_id="row2", severity="low", carrots_inspected=5))
    attempt("bad severity", lambda: Row(row_id="ROW-02", severity="Moderate", carrots_inspected=5))
    # Expect: FAIL, "Moderate" is not in the Literal (case matters)
    attempt("negative count", lambda: Row(row_id="ROW-02", severity="low", carrots_inspected=-1))
    attempt("several errors at once", lambda: Row(row_id="x", severity="??", carrots_inspected=-1))
    # Expect: FAIL, one error per bad field (Pydantic reports all of them)


class CleanSeverity(BaseModel):
    severity: Severity

    @field_validator("severity", mode="before")
    @classmethod
    def normalize(cls, v):
        # mode="before" runs BEFORE the Literal check, so we can clean the input
        return v.strip().lower() if isinstance(v, str) else v


def exp4_field_validator() -> None:
    header("4. @field_validator: clean input before checking it")
    attempt('" Moderate " cleaned', lambda: CleanSeverity(severity=" Moderate "))
    # Expect: OK, severity='moderate'  (this would fix the old "Moderate" vs "moderate" bug)


class Gps(BaseModel):
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)

    @model_validator(mode="after")
    def reject_uninitialized_fix(self):
        # A camera with no GPS lock often reports exactly (0.0, 0.0). That is
        # inside the valid ranges, so ge/le alone can't catch it.
        if self.latitude == 0.0 and self.longitude == 0.0:
            raise ValueError("GPS fix looks uninitialized (0.0, 0.0)")
        return self


def exp5_model_validator_gps() -> None:
    header("5. @model_validator: GPS rules that span two fields")
    attempt("Salinas", lambda: Gps(latitude=36.6777, longitude=-121.6555))
    attempt("lat out of range", lambda: Gps(latitude=91, longitude=-121.6))
    attempt("0.0, 0.0", lambda: Gps(latitude=0.0, longitude=0.0))
    # Expect: FAIL, our custom message (shown under "(model)")
    attempt("NaN latitude", lambda: Gps(latitude=float("nan"), longitude=-121.6))
    # Expect: FAIL because allow_inf_nan=False. Try deleting that option and
    # see what changes (plain float fields accept nan/inf by default in v2).


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row_id: str


class Lenient(BaseModel):
    row_id: str


def exp6_extra_fields() -> None:
    header("6. Unknown fields")
    attempt("default (extras ignored)", lambda: Lenient(row_id="ROW-01", typo_field=1))
    # Expect: OK; typo_field silently dropped. Dangerous when a client misspells a key.
    attempt("extra='forbid'", lambda: Strict(row_id="ROW-01", typo_field=1))
    # Expect: FAIL, extra inputs are not permitted


class ObsUpdate(BaseModel):
    """Same idea as the observation_update model in main.py."""
    description: Optional[str] = None
    severity: Optional[Severity] = None
    carrots_inspected: Optional[int] = Field(default=None, ge=0)


def exp7_partial_updates() -> None:
    header("7. PUT/PATCH: exclude_unset vs exclude_none")
    sent_nothing = ObsUpdate()
    sent_null = ObsUpdate.model_validate({"description": None})
    sent_value = ObsUpdate.model_validate({"severity": "high"})
    print("  sent nothing, exclude_unset :", sent_nothing.model_dump(exclude_unset=True))
    # Expect: {}
    print("  sent null,    exclude_unset :", sent_null.model_dump(exclude_unset=True))
    # Expect: {'description': None}  -> the caller explicitly wants to clear it
    print("  sent null,    exclude_none  :", sent_null.model_dump(exclude_none=True))
    # Expect: {}  -> cannot tell "clear it" from "leave it alone"
    print("  sent severity, exclude_unset:", sent_value.model_dump(exclude_unset=True))
    # Expect: {'severity': 'high'}


class Detection(BaseModel):
    class_name: Literal["discoloration", "disease", "pest", "weed"]
    confidence: float = Field(ge=0, le=1)


class Telemetry(BaseModel):
    """What a field camera might POST (compare UC-7 / MVP-11 in Lab 2)."""
    image_id: str = Field(min_length=1)
    captured_at: AwareDatetime  # must include a timezone
    gps: Gps
    detections: list[Detection] = Field(min_length=1, max_length=500)

    @field_validator("captured_at")
    @classmethod
    def not_in_future(cls, v):
        if v > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ValueError("captured_at is in the future")
        return v


GOOD_PAYLOAD = {
    "image_id": "IMG_0042",
    "captured_at": "2026-10-10T10:15:00-07:00",
    "gps": {"latitude": 36.6777, "longitude": -121.6555},
    "detections": [{"class_name": "discoloration", "confidence": 0.82}],
}


def exp8_nested_payloads() -> None:
    header("8. Nested models, lists, and dates")
    attempt("good payload", lambda: Telemetry.model_validate(GOOD_PAYLOAD))

    bad_conf = {**GOOD_PAYLOAD, "detections": [
        {"class_name": "discoloration", "confidence": 0.9},
        {"class_name": "discoloration", "confidence": 1.7},
    ]}
    attempt("2nd detection confidence 1.7", lambda: Telemetry.model_validate(bad_conf))
    # Expect: FAIL at loc detections.1.confidence (the index tells you which item)

    attempt("naive timestamp", lambda: Telemetry.model_validate(
        {**GOOD_PAYLOAD, "captured_at": "2026-10-10T10:15:00"}))
    # Expect: FAIL, needs timezone info
    attempt("future timestamp", lambda: Telemetry.model_validate(
        {**GOOD_PAYLOAD, "captured_at": "2099-01-01T00:00:00+00:00"}))
    attempt("empty detections", lambda: Telemetry.model_validate(
        {**GOOD_PAYLOAD, "detections": []}))
    attempt("0.0, 0.0 gps", lambda: Telemetry.model_validate(
        {**GOOD_PAYLOAD, "gps": {"latitude": 0, "longitude": 0}}))
    attempt("JSON string input", lambda: Telemetry.model_validate_json(
        '{"image_id": "", "captured_at": "2026-10-10T10:15:00Z",'
        ' "gps": {"latitude": 36.6, "longitude": -121.6}, "detections": []}'))
    # Expect: FAIL, image_id too short AND detections too short (both reported)


class AliasGps(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    latitude: float = Field(alias="lat")
    longitude: float = Field(alias="lon")


def exp9_aliases_and_output() -> None:
    header("9. Aliases and dumping")
    g = AliasGps.model_validate({"lat": 36.6, "lon": -121.6})
    print("  by field name :", g.model_dump())
    print("  by alias      :", g.model_dump(by_alias=True))
    print("  as JSON       :", g.model_dump_json(by_alias=True))
    print("  required keys in the JSON schema FastAPI shows at /docs:",
          Telemetry.model_json_schema()["required"])
    # Expect: ['image_id', 'captured_at', 'gps', 'detections']


class Counter(BaseModel):
    carrots_inspected: int = Field(ge=0)


class CheckedCounter(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    carrots_inspected: int = Field(ge=0)


def exp10_pitfalls() -> None:
    header("10. Pitfalls")
    c = Counter(carrots_inspected=3)
    c.carrots_inspected = -50  # no error: assignment is NOT validated by default
    print("  plain assignment of -50 ->", c)
    # Expect: carrots_inspected=-50 slips through

    def try_checked():
        cc = CheckedCounter(carrots_inspected=3)
        cc.carrots_inspected = -50
        return cc
    attempt("validate_assignment=True", try_checked)
    # Expect: FAIL

    copy = Counter(carrots_inspected=3).model_copy(update={"carrots_inspected": -9})
    print("  model_copy(update=...) with -9 ->", copy)
    # Expect: carrots_inspected=-9; update values are not re-validated.
    # Safer: Counter.model_validate({**old.model_dump(), "carrots_inspected": -9})


CHALLENGES = """
CHALLENGES (try these next)
  1. Add a `row_id` pattern to Telemetry and make 'ROW-7' fail but 'ROW-07' pass.
  2. Add a model_validator that rejects a detection list where every item has
     confidence < 0.1 (a likely garbage image).
  3. Make severity case-insensitive using exp4's trick, then check the old
     sample JSON (carrot_field_sample.json) loads without errors.
  4. Wrap Telemetry in a FastAPI endpoint, POST a bad payload with curl or
     /docs, and read the 422 body. Compare it to e.errors() from exp8.
  5. Time model_validate on 500 detections (timeit). Is validation a
     bottleneck for your telemetry endpoint?
"""


if __name__ == "__main__":
    for experiment in (
        exp1_coercion, exp2_strict_mode, exp3_field_rules, exp4_field_validator,
        exp5_model_validator_gps, exp6_extra_fields, exp7_partial_updates,
        exp8_nested_payloads, exp9_aliases_and_output, exp10_pitfalls,
    ):
        experiment()
    print(CHALLENGES)
