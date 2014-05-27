package org.jabe.neverland.activity;

import org.jabe.neverland.R;
import org.jabe.neverland.view.FingerPaintView;

import android.app.Activity;
import android.os.Bundle;

public class HandWriterActivity extends Activity {

	FingerPaintView fpv;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.act_handwriter_main);
		fpv = (FingerPaintView) findViewById(R.id.fingerPaintView);
	}
}
