# Acme Corp — Technical FAQ

## Account & Login

### How do I reset my password?

Go to the login page and click "Forgot Password". Enter your registered email address and we will send you a reset link within 5 minutes. The link is valid for 1 hour. If you don't receive the email, check your spam folder or contact support.

### Can I have multiple accounts?

No. Each email address can only be associated with one account. If you need a team account, look into our Business Plan which supports multiple users under one billing account.

### How do I delete my account?

Account deletion can be requested from Settings → Account → Delete Account. Note that deletion is permanent and cannot be undone. All data associated with your account will be removed within 30 days.

---

## Billing

### What payment methods do you accept?

We accept Visa, Mastercard, American Express, PayPal, and bank transfers for annual plans. Cryptocurrency is not accepted.

### How do I download my invoice?

Log in → Settings → Billing → Invoice History. Invoices are available in PDF format for the last 24 months.

### What happens if my payment fails?

If a payment fails, we will retry it after 3 days and again after 7 days. After two failed attempts, your account will be downgraded to the free tier. You will receive email notifications at each step.

---

## Integrations

### Does Acme support Slack notifications?

Yes. Go to Settings → Integrations → Slack and follow the OAuth flow. You can configure which events trigger notifications.

### Is there an API?

Yes. Our REST API is documented at https://docs.acmecorp.example.com/api. Authentication uses Bearer tokens which you can generate in Settings → API Keys.

### What are the API rate limits?

Free tier: 100 requests/hour. Pro tier: 5,000 requests/hour. Enterprise: unlimited (subject to fair use).

---

## Data & Privacy

### Where is my data stored?

All data is stored in AWS US-East-1 by default. EU customers can request EU data residency (AWS eu-west-1) — contact sales for this option.

### Is my data encrypted?

Yes. All data is encrypted at rest (AES-256) and in transit (TLS 1.2+). We do not share your data with third parties.

### How long is data retained?

Active account data is retained indefinitely. After account deletion, data is purged within 30 days. Backup data may persist for up to 90 days.
