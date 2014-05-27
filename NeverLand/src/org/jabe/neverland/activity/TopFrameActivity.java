package org.jabe.neverland.activity;

import org.jabe.neverland.R;
import org.jabe.neverland.view.ShaderTextView;

import android.app.Activity;
import android.graphics.Rect;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.WindowManager;
import android.view.WindowManager.LayoutParams;
import android.widget.Button;

/**
 */
public class TopFrameActivity extends Activity {
	
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.act_top_frame);

		Button button = (Button) findViewById(R.id.bt);
		
		button.setOnClickListener(onclick);
	}
	
	private ShaderTextView tv = null;
	
	OnClickListener onclick = new OnClickListener() {
		
		@Override
		public void onClick(View v) {
			if(tv != null && tv.isShown()){
				WindowManager wm = (WindowManager)getApplicationContext().getSystemService(TopFrameActivity.this.WINDOW_SERVICE);
				wm.removeView(tv);
			}
			show();
			
		}
	};
	
	
	private void show(){
		Rect frame = new Rect();  
		getWindow().getDecorView().getWindowVisibleDisplayFrame(frame);  
		ShaderTextView.TOOL_BAR_HIGH = frame.top;  
		
		WindowManager wm = (WindowManager)getApplicationContext().getSystemService(WINDOW_SERVICE);
		WindowManager.LayoutParams params = ShaderTextView.params;
		
		params.type = WindowManager.LayoutParams.TYPE_SYSTEM_ALERT | WindowManager.LayoutParams.TYPE_SYSTEM_OVERLAY;
		params.flags = LayoutParams.FLAG_NOT_TOUCH_MODAL | LayoutParams.FLAG_NOT_FOCUSABLE;
		
		params.width = WindowManager.LayoutParams.FILL_PARENT;
		params.height = WindowManager.LayoutParams.WRAP_CONTENT;
		params.alpha = 80;
		
		params.gravity = Gravity.LEFT|Gravity.TOP;
		params.x = 0;
		params.y = 0;
	        
		tv = new ShaderTextView(TopFrameActivity.this);
		wm.addView(tv, params);
	}

	@Override
	protected void onDestroy() {
		tryDismiss();
		super.onDestroy();
	}
	
	@Override
	public void onBackPressed() {
		tryDismiss();
		super.onBackPressed();
	}

	private void tryDismiss() {
		WindowManager wm = (WindowManager)getApplicationContext().getSystemService(WINDOW_SERVICE);
		if(tv != null && tv.isShown()){
			try {
				wm.removeView(tv);
				tv = null;
			} catch (Exception e) {
			}
		}
	}
	
	
	
}