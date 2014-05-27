package org.jabe.neverland.activity;

import org.jabe.neverland.R;
import org.jabe.neverland.view.MyClockView;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.KeyEvent;

public class ColClockAdjustActivity extends Activity {
	/** Called when the activity is first created. */

	MyClockView myClockView;

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		init();
	}

	public void init() {
		myClockView = new MyClockView(this);
		Bitmap clockBitmap = BitmapFactory.decodeResource(getResources(),
				R.drawable.clock);
		Bitmap hourBitmap = BitmapFactory.decodeResource(getResources(),
				R.drawable.hour);
		Bitmap minuteBitmap = BitmapFactory.decodeResource(getResources(),
				R.drawable.minute);
		myClockView.init(clockBitmap, hourBitmap, minuteBitmap);
		myClockView.setCenter(163, 163); // 本例时钟中心点相对于视图坐标为（163， 163）
		myClockView.setPointOffset(20, 20); // 12点钟方向指针向下偏移20像素
		setContentView(myClockView);
	}

	public boolean onKeyDown(int keyCode, KeyEvent event) {

		myClockView.onKeyDown(keyCode, event);

		return super.onKeyDown(keyCode, event);
	}
}