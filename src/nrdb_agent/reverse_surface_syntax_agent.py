import json

from .reverse_surface_agent import ReverseSurfaceAgent, SURFACE_FORMAT


SURFACE_SYNTAX_REPAIR_INSTRUCTIONS = """You are the structural repair phase for NRDB Miyako surface realization.

The NRDB annotation is FROZEN. Repair ONLY the segmented trsc2 surface so that it obeys the annotation's realization syntax.

NRDB annotation syntax is binding:
- A SPACE separates phrases.
- A HYPHEN (-) separates independently realized annotation segments and therefore separate surface segments.
- A SEMICOLON (;) conflates multiple annotation atoms inside ONE annotation segment. It NEVER creates another surface segment.
- Therefore every annotation phrase and surface phrase must have exactly the same number of hyphen-delimited segments.
- Never realize semicolon-conflated atoms as independent hyphenated morphemes.

Examples:
- annotation `火un-acc-foc` has 3 surface slots.
- annotation `消kv;cvb` has 1 surface slot: an output like `kjaʃ-ti` is structurally wrong because it invents a boundary inside a conflated segment.
- annotation `眠nv;cvb-foc-ipf` has 3 surface slots: `[眠nv;cvb] [foc] [ipf]`.

Preserve the intended lexical choices and grammar. Do not change IDs, phrase structure, semantic content, or n: layer membership. If the invalid surface over-split a conflated annotation segment, re-realize that complete conflated segment as ONE trsc2 surface segment. Do not merely reinterpret `;` as `-`.

Return exactly one JSON object:
{"segmented":"...","confidence":0.0,"evidence":{"note":"brief","ids_realized":[],"fallback_ids":[],"example_sentence_ids":[]}}
"""


def annotation_surface_skeleton(annotation):
	phrases = str(annotation or "").strip().split()
	return [phrase.split("-") if phrase else [] for phrase in phrases]


def surface_alignment_error(segmented, annotation):
	seg_phrases = str(segmented or "").strip().split()
	ann_phrases = str(annotation or "").strip().split()
	if len(seg_phrases) != len(ann_phrases):
		return "surface has {} phrase(s), frozen annotation has {}".format(len(seg_phrases), len(ann_phrases))
	for phrase_index, (seg_phrase, ann_phrase) in enumerate(zip(seg_phrases, ann_phrases), start=1):
		surfaces = seg_phrase.split("-") if seg_phrase else []
		labels = ann_phrase.split("-") if ann_phrase else []
		if any(not value for value in surfaces):
			return "surface phrase {} contains an empty hyphen-delimited segment".format(phrase_index)
		if len(surfaces) != len(labels):
			return "surface phrase {} has {} segment(s), frozen annotation has {}; semicolon atoms stay inside one surface segment".format(
				phrase_index, len(surfaces), len(labels),
			)
	return None


class SyntaxAwareReverseSurfaceAgent(ReverseSurfaceAgent):
	"""Reverse surface agent that enforces NRDB hyphen/semicolon realization syntax."""

	def _repair_surface_syntax(self, item, job, id_result, surface_result, error):
		annotation = id_result.get("annotation") or ""
		payload = {
			"japanese": str(item.get("translation_jp") or "").strip(),
			"frozen_annotation": annotation,
			"annotation_surface_skeleton": annotation_surface_skeleton(annotation),
			"invalid_segmented": surface_result.get("segmented") or "",
			"structural_error": error,
			"target_dialect_ids": job.get("target_dialect_ids") or [int(item["dialect_id"])],
		}
		self.progress("  surface-syntax: invalid realization: {}".format(error))
		self.progress("  surface-syntax: one constrained repair pass")
		response = self._create_response(
			[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
			SURFACE_SYNTAX_REPAIR_INSTRUCTIONS, tools=[], max_output_tokens=900, text_format=SURFACE_FORMAT,
		)
		candidate = self._parse_surface(response.output_text)
		candidate_error = surface_alignment_error(candidate.get("segmented"), annotation)
		if candidate_error:
			raise ValueError("reverse surface violates frozen annotation syntax after repair: {}".format(candidate_error))
		candidate["confidence"] = min(float(surface_result.get("confidence", 0.0)), float(candidate.get("confidence", 0.0)))
		candidate.setdefault("evidence", {})["syntax_repaired"] = True
		candidate["evidence"]["syntax_error"] = error
		candidate["evidence"]["pre_repair_segmented"] = surface_result.get("segmented") or ""
		candidate["evidence"]["pre_repair_surface_evidence"] = surface_result.get("evidence", {})
		self.progress("  surface-syntax: repaired={!r}".format(candidate.get("segmented") or ""))
		return candidate

	def _realize_surface(self, item, job, id_result):
		surface = super()._realize_surface(item, job, id_result)
		error = surface_alignment_error(surface.get("segmented"), id_result.get("annotation"))
		if error:
			return self._repair_surface_syntax(item, job, id_result, surface, error)
		surface.setdefault("evidence", {})["syntax_repaired"] = False
		return surface
