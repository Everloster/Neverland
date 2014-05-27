package org.jabe.neverland.plugin.impl;

import org.jabe.neverland.R;

import android.content.Intent;
import android.os.Bundle;

public class TestPluginActivity extends PluginImpl {

	@Override
	public void onPluginDestroy() {
		// TODO Auto-generated method stub

	}

	@Override
	public void onPluginPause() {
		// TODO Auto-generated method stub

	}

	@Override
	public void onPluginResume() {
		// TODO Auto-generated method stub

	}

	@Override
	public void onPluginStart() {
		// TODO Auto-generated method stub

	}

	@Override
	public void onPluginStop() {
		// TODO Auto-generated method stub

	}

	@Override
	public void onPluginActivityResult(int requestCode, int resultCode,
			Intent data) {
		
	}
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
	}
}
