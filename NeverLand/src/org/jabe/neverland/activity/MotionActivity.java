package org.jabe.neverland.activity;

import org.jabe.neverland.R;
import org.jabe.neverland.util.SmileyParser;
import org.jabe.neverland.view.DefaultFaceView;
import org.jabe.neverland.view.FaceView.ChooseFaceListener;

import android.app.Activity;
import android.os.Bundle;
import android.widget.EditText;

public class MotionActivity extends Activity {
	
	private DefaultFaceView emotionLayout;
	private EditText mEditText;
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.act_motion_main);
		setUpViews();
	}
	private void setUpViews() {
		mEditText = (EditText) findViewById(R.id.motion_edit);
		emotionLayout = (DefaultFaceView) findViewById(R.id.motion_view);
		emotionLayout.setFaceListener(new ChooseFaceListener() {
			@Override
			public void onChoosed(String text) {
				mEditText.append(SmileyParser.getInstance(MotionActivity.this).addSmileySpans(text));
			}
		});
	}
}
