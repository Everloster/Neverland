package org.jabe.neverland.activity;

import org.jabe.neverland.R;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.provider.Settings;
import android.view.ActionProvider;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.Window;
import android.widget.ImageButton;
import android.widget.Toast;

@SuppressLint("NewApi")
public class MenuTestActivity extends Activity {

	private static final Intent sSettingsIntent = new Intent(
			Settings.ACTION_SETTINGS);

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		super.onCreateOptionsMenu(menu);
		getMenuInflater().inflate(R.menu.test_all_menu, menu);
		return true;
	}
	
	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		// TODO Auto-generated method stub
//		Toast.makeText(this, "fuck~", Toast.LENGTH_SHORT).show();
		if (item.getItemId() == R.id.action_setting) {
			this.startActivity(sSettingsIntent);
		}
		return true;
	}
	
	public static class CustomActionProvider extends ActionProvider {
		private final Context mContext;

		public CustomActionProvider(Context context) {
			super(context);
			mContext = context;
		}

		/**
		 * {@inheritDoc}
		 */
		@Override
		public View onCreateActionView() {
			// Inflate the action view to be shown on the action bar.
			LayoutInflater layoutInflater = LayoutInflater.from(mContext);
			View view = layoutInflater.inflate(
					R.layout.action_bar_settings_action_provider, null);
			ImageButton button = (ImageButton) view.findViewById(R.id.button);
			// Attach a click listener for launching the system settings.
			button.setOnClickListener(new View.OnClickListener() {
				@Override
				public void onClick(View v) {
					Toast.makeText(mContext, "fuck1", Toast.LENGTH_SHORT).show();
				}
			});
			return view;
		}
		
		@Override
        public boolean onPerformDefaultAction() {
            // This is called if the host menu item placed in the overflow menu of the
            // action bar is clicked and the host activity did not handle the click.
			Toast.makeText(mContext, "fuck2", Toast.LENGTH_SHORT).show();
            return true;
        }

	}
}
