from nrdb_agent.reverse_surface_critic_agent import SurfaceCriticReverseAgent
from nrdb_agent.reverse_surface_syntax_agent import (
	SyntaxAwareReverseSurfaceAgent,
	annotation_surface_skeleton,
	surface_alignment_error,
)


def test_semicolon_is_one_surface_slot():
	assert annotation_surface_skeleton("消kv;cvb") == [["消kv;cvb"]]
	assert surface_alignment_error("kjaːʃi", "消kv;cvb") is None
	assert "semicolon atoms stay inside one surface segment" in surface_alignment_error("kjaʃ-ti", "消kv;cvb")


def test_hyphen_defines_surface_slots():
	assert annotation_surface_skeleton("火un-acc-foc") == [["火un", "acc", "foc"]]
	assert surface_alignment_error("umat-tsu-du", "火un-acc-foc") is None
	assert surface_alignment_error("umat-tsudu", "火un-acc-foc") is not None


def test_conflation_and_independent_suffixes_combine_correctly():
	assert annotation_surface_skeleton("眠nv;cvb-foc-ipf") == [["眠nv;cvb", "foc", "ipf"]]
	assert surface_alignment_error("nivvi-du-u", "眠nv;cvb-foc-ipf") is None
	assert surface_alignment_error("nivvi-cvb-du-u", "眠nv;cvb-foc-ipf") is not None


def test_surface_critic_inherits_syntax_guard():
	assert issubclass(SurfaceCriticReverseAgent, SyntaxAwareReverseSurfaceAgent)
