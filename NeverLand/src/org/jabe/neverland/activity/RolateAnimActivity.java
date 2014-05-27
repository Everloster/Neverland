package org.jabe.neverland.activity;

import org.jabe.neverland.R;
import org.jabe.neverland.view.MyImageView;

import android.app.Activity;
import android.os.Bundle;
import android.widget.Toast;

public class RolateAnimActivity extends Activity {
    /** Called when the activity is first created. */
	MyImageView joke;
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.act_rolate_imageview);
        
        joke=(MyImageView) findViewById(R.id.c_joke);
        joke.setOnClickIntent(new MyImageView.OnViewClick() {
			
			@Override
			public void onClick() {
				// TODO Auto-generated method stub
				Toast.makeText(RolateAnimActivity.this, "事件触发", 1000).show();
			}
		});
    }
}