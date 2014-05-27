package org.jabe.neverland.activity;

import org.jabe.neverland.view.PageTurningView;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.Window;
import android.view.WindowManager;

public class PageTurnActivity extends Activity {

    PageTurningView ptview;
	AlertDialog.Builder builder;
	@Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
      //设置横屏显示
        //setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
//        if(this.getResources().getConfiguration().orientation ==Configuration.ORIENTATION_PORTRAIT){
//        	   setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
//        	}
        
		requestWindowFeature(Window.FEATURE_NO_TITLE);
		getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
				WindowManager.LayoutParams.FLAG_FULLSCREEN);
		ptview=new PageTurningView(this);
        setContentView(ptview);
        builder=new AlertDialog.Builder(this);
    }

	@Override

	public boolean onKeyDown(int keyCode, KeyEvent event) {
	    return super.onKeyDown(keyCode, event);
	}
}
