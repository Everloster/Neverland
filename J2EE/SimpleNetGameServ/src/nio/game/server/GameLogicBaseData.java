package nio.game.server;

public abstract class GameLogicBaseData {

	public int m_nnID;
	
	public GameLogicBaseData(int nnID){
		this.m_nnID = nnID;
	}
	
	public abstract void OnCreate();
	
	public abstract void OnDestroy();
	
	public abstract void OnSave();
	
	public void Save(){
		this.OnSave();
	}
	
}
