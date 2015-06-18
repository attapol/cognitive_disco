package types;

import java.util.Arrays;
import java.util.HashSet;

import cc.mallet.types.InstanceList;

public class Sense {

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
	
	private String sense;
	private String[] splitSense;
	private String[] dimensions;
	
	public Sense(String sense){
		this.sense = sense;
		this.splitSense = sense.split("\\.");
	}
	
	public String getRawSense() {
		return sense;
	}

	/*
	 * 
	 */
	public String getTopLevelLabel(){
		return splitSense[0];
	}
	
	public void setDimensions(String[] dimensions) {
		this.dimensions = dimensions;
	}
	
	public String[] getDimensions(){
		return this.dimensions;
	}
	
	public static InstanceList convertToTopLevel(InstanceList data){
		return null;
	}
	
}
