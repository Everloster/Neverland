package org.jabe.neverland.smallboy;

public class BoyPosition {
	private int x = 0;
	private int y = 0;
	public int getX() {
		return x;
	}
	public void setX(int x) {
		this.x = x;
	}
	public int getY() {
		return y;
	}
	public void setY(int y) {
		this.y = y;
	}
	public BoyPosition(int x, int y) {
		super();
		this.x = x;
		this.y = y;
	}
	
	public BoyPosition() {
		super();
		this.x = 0;
		this.y = 0;
	}
}
