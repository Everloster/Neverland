package net.game.pojo;

import nio.game.server.GameLogicBaseData;

public class GameData extends GameLogicBaseData{
	
	public String m_strName;	//Name in game
	public int m_nX;	//x position
	public int m_nY;	//y position
	
	public GameData(int nnID) {
		super(nnID);
	}
	
	@Override
	public void OnCreate() {
		this.m_nX = 100;
		this.m_nY = 100;
	}
	
	@Override
	public void OnDestroy() {
		
	}

	@Override
	public void OnSave() {
		
	}
}
