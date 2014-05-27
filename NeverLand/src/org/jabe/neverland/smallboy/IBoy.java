package org.jabe.neverland.smallboy;

import android.view.View;
import android.view.WindowManager;

public interface IBoy {
	
	public static final int POSITION_STATUS_LEFT = 1001;
	public static final int POSITION_STATUS_RIGHT = POSITION_STATUS_LEFT + 1;
	public static final int POSITION_STATUS_TOP = POSITION_STATUS_LEFT + 2;
	public static final int POSITION_STATUS_BOTTOM = POSITION_STATUS_LEFT + 3;
	
	public WindowManager.LayoutParams getWindowLayoutParams();
	
	public void setWindowLayoutParams(WindowManager.LayoutParams layoutParams);
	
	public void setToolBarHight(int hight);
	
	public int getToolBarHight();
	
	public void setCallback(IBoyAction action);
	
	public View getRootView();
	
	public View getBoyView();
	
	public void displayMessage(String message);
	
	public void init();
	
	public void destroy();
	
	public void dismiss();
	
	public void showAnimation();
	
}
