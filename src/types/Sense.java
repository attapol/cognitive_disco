package types;

import java.util.Arrays;
import java.util.HashSet;

public class Sense {

	public static final String NULL_SENSE = "**NULL**";
	public static final HashSet<String> FINEST_SENSES = new HashSet<String>(Arrays.asList(
		"Comparison.Concession.Contra-expectation",
		"Comparison.Concession.Expectation",
		"Comparison.Contrast.Juxtaposition",
		"Comparison.Contrast.Opposition",
		"Comparison.Pragmatic concession",
		"Comparison.Pragmatic contrast",
		"Contingency.Cause.Reason",
		"Contingency.Cause.Result",
		"Contingency.Condition.Factual past",
		"Contingency.Condition.Factual present",
		"Contingency.Condition.General",
		"Contingency.Condition.Hypothetical",
		"Contingency.Condition.Unreal past",
		"Contingency.Condition.Unreal present",
		"Contingency.Pragmatic cause.Justification",
		"Contingency.Pragmatic condition.Implicit assertion",
		"Contingency.Pragmatic condition.Relevance",
		"Expansion.Alternative.Chosen alternative",
		"Expansion.Alternative.Conjunctive",
		"Expansion.Alternative.Disjunctive",
		"Expansion.Restatement.Equivalence",
		"Expansion.Restatement.Generalization",
		"Expansion.Restatement.Specification",
		"Temporal.Asynchronous.Precedence",
		"Temporal.Asynchronous.Succession",
		"Temporal.Synchrony"
	));

	public static final HashSet<String> VALID_SENSES = new HashSet<String>(Arrays.asList(
		"Comparison",
		"Comparison.Concession",
		"Comparison.Concession.Contra-expectation",
		"Comparison.Concession.Expectation",
		"Comparison.Contrast",
		"Comparison.Contrast.Juxtaposition",
		"Comparison.Contrast.Opposition",
		"Comparison.Pragmatic concession",
		"Comparison.Pragmatic contrast",
		"Contingency",
		"Contingency.Cause",
		"Contingency.Cause.Reason",
		"Contingency.Cause.Result",
		"Contingency.Condition",
		"Contingency.Condition.Factual past",
		"Contingency.Condition.Factual present",
		"Contingency.Condition.General",
		"Contingency.Condition.Hypothetical",
		"Contingency.Condition.Unreal past",
		"Contingency.Condition.Unreal present",
		"Contingency.Pragmatic cause",
		"Contingency.Pragmatic cause.Justification",
		"Contingency.Pragmatic condition",
		"Contingency.Pragmatic condition.Implicit assertion",
		"Contingency.Pragmatic condition.Relevance",
		"Expansion",
		"Expansion.Alternative",
		"Expansion.Alternative.Chosen alternative",
		"Expansion.Alternative.Conjunctive",
		"Expansion.Alternative.Disjunctive",
		"Expansion.Conjunction",
		"Expansion.Exception",
		"Expansion.Instantiation",
		"Expansion.List",
		"Expansion.Restatement",
		"Expansion.Restatement.Equivalence",
		"Expansion.Restatement.Generalization",
		"Expansion.Restatement.Specification",
		"Temporal",
		"Temporal.Asynchronous",
		"Temporal.Asynchronous.Precedence",
		"Temporal.Asynchronous.Succession",
		"Temporal.Synchrony"
	));
	
	public static boolean isValidSense(String sense){
		return VALID_SENSES.contains(sense);
	}
	
	public static boolean isFinestSense(String sense){
		return FINEST_SENSES.contains(sense);
	}
	
	private String sense;
	private String[] splitSense;
	private String[] dimensions;
	
	public Sense(String sense){
		this.sense = sense;
		this.splitSense = sense.split("\\.");
	}

	/*
	 *  Returns the raw sense which can be anywhere on the sense hierarchy
	 */
	public String getRawSense() {
		return sense;
	}
	
	public String getLabel(LabelType labelType) {
		switch (labelType) {
		case TOP_LEVEL:
			return getTopLevelLabel();
		case CONLL:
			return getCoNLLLabel();
		case SCHEME_B:
			return getSchemeBLabel();
		}
		return null;
	}

	/*
	 *  Returns the top level sense
	 */
	public String getTopLevelLabel() {
		return splitSense[0];
	}

	public static String getTopLevelLabel(String originalLabel) {
		Sense s = new Sense(originalLabel);
		return s.getTopLevelLabel();
	}

	/*
	 *  Returns the CoNLL sense including EntRel
	 */
	public String getCoNLLLabel() {
		int numSplit = splitSense.length;
		if (sense.equals("EntRel")) return sense;
		if (numSplit > 1){
			if (sense.equals("Expansion.Alternative.Chosen alternative")) {
				return sense;
			}else if ((splitSense[0].equals("Contingency") && splitSense[1].equals("Cause")) || 
					(splitSense[0].equals("Temporal") && splitSense[1].equals("Asynchronous"))) {
				return numSplit == 3 ? sense : NULL_SENSE;
			}
			else if (splitSense[1].equals("Pragmatic cause")) {
				return "Contingency.Cause.Reason";
			} else if (splitSense[1].equals("Pragmatic condition")) {
				return "Contingency.Condition";
			} else if (splitSense[1].equals("Pragmatic contrast")) {
				return "Comparison.Contrast";
			} else if (splitSense[1].equals("Pragmatic concession")) {
				return "Comparison.Concession";
			} else if (splitSense[1].equals("List")) {
				return "Expansion.Conjunction";
			} else
				return splitSense[0]+"."+splitSense[1];
		}
		return NULL_SENSE;
	}
	
	public static String getCoNLLLabel(String rawSense) {
		Sense s = new Sense(rawSense);
		return s.getCoNLLLabel();
	}
	
	public String getSchemeBLabel() {
		int numSplit = splitSense.length;
		if (numSplit > 1 && !splitSense[1].equals("Condition") &&
				!splitSense[1].equals("Pragmatic condition") &&
				!splitSense[1].equals("Pragmatic contrast") &&
				!splitSense[1].equals("Pragmatic concession") &&
				!splitSense[1].equals("Exception")) {
			return splitSense[0]+"."+splitSense[1];
		}
		return NULL_SENSE;
	}
	
	public static String getSchemeBLabel(String rawSense) {
		Sense s = new Sense(rawSense);
		return s.getSchemeBLabel();
	}
	
	public boolean isAtLeastSecondLevel() {
		return splitSense.length > 1;
	}
	
	public boolean isFinestSense() {
		return isFinestSense(this.sense);
	}
	
	public String getLevel2Label(){
		return isAtLeastSecondLevel() ? splitSense[1] : "";
	}
	
	public void setDimensions(String[] dimensions) {
		this.dimensions = dimensions;
	}
	
	public String[] getDimensions(){
		return this.dimensions;
	}
	
	
}
