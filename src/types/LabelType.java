package types;

public enum LabelType {
	TOP_LEVEL ("Top Level"),
	CONLL ("CoNLL Label"),
	SCHEME_B ("Modified Level 2");

	private final String name;
	LabelType(String name){
		this.name = name;
	}
	public String toString() {
		return this.name;
	}
}
