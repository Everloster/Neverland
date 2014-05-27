package org.jabe.neverland.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Shader;
import android.graphics.Shader.TileMode;
import android.graphics.Typeface;
import android.os.Handler;
import android.util.Log;
import android.view.MotionEvent;
import android.view.WindowManager;
import android.widget.TextView;

public class ShaderTextView extends TextView {
	private final String TAG = ShaderTextView.class.getSimpleName();

	public static int TOOL_BAR_HIGH = 0;
	public static WindowManager.LayoutParams params = new WindowManager.LayoutParams();
	private float startX;
	private float startY;
	private float x;
	private float y;

	private String text;

	WindowManager wm = (WindowManager) getContext().getApplicationContext()
			.getSystemService(getContext().WINDOW_SERVICE);

	public ShaderTextView(Context context) {
		super(context);
		text = "世上只有老婆好~~~~~~~";
		this.setBackgroundColor(Color.argb(90, 150, 150, 150));
		handler = new Handler();
		// handler.post(update);
	}

	@Override
	public boolean onTouchEvent(MotionEvent event) {
		x = event.getRawX();
		y = event.getRawY() - TOOL_BAR_HIGH;
		Log.d(TAG, "------X: " + x + "------Y:" + y);

		switch (event.getAction()) {
		case MotionEvent.ACTION_DOWN:
			startX = event.getX();
			startY = event.getY();
			break;
		case MotionEvent.ACTION_MOVE:
			updatePosition();
			break;
		case MotionEvent.ACTION_UP:
			updatePosition();
			startX = startY = 0;
			break;
		}

		return true;
	}

	@Override
	protected void onDraw(Canvas canvas) {
		// TODO Auto-generated method stub
		super.onDraw(canvas);
		float1 += 0.001f;
		float2 += 0.001f;

		if (float2 > 1.0) {
			float1 = 0.0f;
			float2 = 0.01f;
		}
		this.setText("");
		float len = this.getTextSize() * text.length();
		Shader shader = new LinearGradient(0, 0, len, 0, new int[] {
				Color.YELLOW, Color.RED }, new float[] { float1, float2 },
				TileMode.CLAMP);
		Paint p = new Paint();
		p.setShader(shader);
		p.setTypeface(Typeface.DEFAULT_BOLD);
		canvas.drawText(text, 0, 10, p);
	}

	private Runnable update = new Runnable() {
		public void run() {
			ShaderTextView.this.update();
			handler.postDelayed(update, 5);
		}
	};

	private void update() {
		postInvalidate(); // ���½���
	}

	private float float1 = 0.0f;
	private float float2 = 0.01f;

	private Handler handler;

	private void updatePosition() {
		params.x = (int) (x - startX);
		params.y = (int) (y - startY);
		wm.updateViewLayout(this, params);
	}
}
